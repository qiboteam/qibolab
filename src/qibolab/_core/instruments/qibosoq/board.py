"""QICK-Qibosoq interface.

This version is supporting qibosoq 0.2 (namely the tprocv2).
It also adds support for multiple synchronized boards.
"""

import re
from collections import defaultdict
from dataclasses import replace
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import qibosoq.components.base as rfsoc
from pydantic import BaseModel, Field
from qibo.config import log
from qibosoq import client
from scipy.constants import micro, nano

from qibolab._core.components.channels import AcquisitionChannel, Channel, DcChannel
from qibolab._core.components.configs import AcquisitionConfig, Config, DcConfig
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import ChannelId, Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.pulses import Pulse
from qibolab._core.pulses.pulse import PulseId
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers, Parameter

from .convert import convert, convert_units_sweeper

__all__ = ["RFSoC", "RFSoCConfig"]


class BoardSettings(BaseModel):
    """Connection and synchronization settings for one RFSoC board."""

    host: str = Field(min_length=1)
    port: int = Field(default=6000, ge=1, le=65535)
    delay: float = 0
    timeout: float = -1


class RFSoCConfig(Config):
    """Qibolab configuration for a coordinated group of RFSoC boards.

    The position of each entry in ``boards`` is its board index. In a
    multi-board configuration, board 0 is the master and every other board is
    a slave. A configuration containing one board uses immediate start mode.
    """

    kind: Literal["qibosoq"] = "qibosoq"
    boards: list[BoardSettings] = Field(min_length=1)
    ro_time_of_flight: int = 200
    soft_avgs: int = 1
    max_retries: int | None = Field(default=None, ge=1)


class RFSoC(Controller):
    """Instrument controlling RFSoC FPGAs."""

    # ``address`` is retained for compatibility with old single-board
    # platforms. Multi-board platforms load their endpoints from RFSoCConfig.
    address: str = ""
    _sampling_rate: float = 10e9
    cfg: rfsoc.Config = Field(default_factory=rfsoc.Config)
    """Configuration dictionary required for pulse execution."""
    config: str = "rfsoc/config"
    """Key of the RFSoCConfig entry in the platform configuration mapping."""

    @property
    def sampling_rate(self):
        """Sampling rate of RFSoC."""
        return self._sampling_rate

    def connect(self):
        """Empty method to comply with Instrument interface."""

    def disconnect(self):
        """Empty method to comply with Instrument interface."""

    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[PulseId, Result]:
        """Play a pulse sequence and retrieve feedback."""
        results = {}

        for seq in sequences:
            _validate_input_command(seq, options, sweepers)
            board_settings = self._board_settings(configs)
            _validate_board_channels(self.channels, len(board_settings.boards))

            # A hardware sweeper would make only the board owning the swept
            # channel iterate. Until qibosoq exposes a synchronized no-op
            # sweeper, coordinated executions use software sweepers so that
            # every sweep point starts all boards together.
            fw = (
                _firmware_loops(seq, sweepers, configs, self.channels)
                if len(board_settings.boards) == 1
                else 0
            )
            res = self._sweep(configs, seq, sweepers, len(sweepers) - fw, options, {})
            results |= _reshape_sweep_results(res, sweepers, options, fw)

        return results

    def _sweep(
        self,
        configs: dict[str, Config],
        sequence: PulseSequence,
        sweepers: list[ParallelSweepers],
        software: int,
        options: ExecutionParameters,
        updates: dict,
    ) -> dict[PulseId, Result]:
        """Execute a sweep of an arbitrary number of sweepers via recursion."""
        # If there are no software sweepers send experiment.
        # Last layer for recursion.
        if software == 0:
            if len(dict(updates)) > 0:
                log.info(f"Executing sequence with updates: {dict(updates)}")
            return self._play(configs, sequence, sweepers, options, updates)

        # use a default dictionary, merging existing values
        updates = defaultdict(dict) | ({} if updates is None else updates)

        parsweep = sweepers[0]
        results = {}
        for values in zip(*(s.values for s in parsweep)):
            # update all parallel sweepers with the respective values
            for sweeper, value in zip(parsweep, values):
                if sweeper.pulses is not None:
                    for pulse in sweeper.pulses:
                        updates[pulse.id].update({sweeper.parameter.name: value})
                if sweeper.channels is not None:
                    for channel in sweeper.channels:
                        updates[channel].update({sweeper.parameter.name: value})

            res = self._sweep(
                configs, sequence, sweepers[1:], software - 1, options, updates
            )
            results = _merge_sweep_results(results, res)

        return results

    def _play(
        self,
        configs: dict[str, Config],
        sequence: PulseSequence,
        sweepers: list[ParallelSweepers],
        options: ExecutionParameters,
        updates: dict,
    ) -> dict[PulseId, Result]:
        """Execute pulse sequence or on-hardware sweep."""
        results = {}

        self.cfg.average = (
            options.acquisition_type is not AcquisitionType.DISCRIMINATION
            and options.averaging_mode is AveragingMode.CYCLIC
        )
        opcode = (
            rfsoc.OperationCode.EXECUTE_PULSE_SEQUENCE_RAW_V2
            if options.acquisition_type is AcquisitionType.RAW
            else rfsoc.OperationCode.EXECUTE_PULSE_SEQUENCE_V2
        )
        board_results = self._execute(
            _update_configs(configs, updates),
            _update_sequence(sequence, updates),
            sweepers,
            opcode,
            options,
        )

        board_count = len(board_results)
        for board, (toti, totq) in enumerate(board_results):
            board_acquisitions = [
                (ch, acq)
                for ch, acq in sequence.acquisitions
                if _channel_board(ch, board_count) == board
            ]
            acq_chs = np.unique([ch for ch, _ in board_acquisitions])

            if len(toti) != len(acq_chs) or len(totq) != len(acq_chs):
                raise RuntimeError(
                    "Unexpected acquisition-channel count returned by board "
                    f"{board}: expected {len(acq_chs)}, got "
                    f"I={len(toti)}, Q={len(totq)}."
                )

            for idx, this_ch in enumerate(acq_chs):
                this_ch_acq = [
                    (ch, acq) for ch, acq in board_acquisitions if ch == this_ch
                ]
                for i, q, (ch, acq) in zip(toti[idx], totq[idx], this_ch_acq):
                    if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                        config = cast(AcquisitionConfig, configs[ch])
                        angle, threshold = config.iq_angle, config.threshold
                        assert angle is not None and threshold is not None
                        result = _classify_shots(
                            np.array(i), np.array(q), angle, threshold
                        )

                        if options.averaging_mode is AveragingMode.CYCLIC:
                            result = np.mean(result, axis=0)

                    else:
                        result = np.stack([i, q], axis=-1)
                    results[acq.id] = result

        return results

    def _board_settings(self, configs: dict[str, Config]) -> RFSoCConfig:
        """Load board settings, with a legacy single-board fallback."""
        settings = configs.get(self.config)
        if settings is not None:
            if not isinstance(settings, RFSoCConfig):
                raise TypeError(
                    f"Configuration {self.config!r} must be an RFSoCConfig, "
                    f"not {type(settings).__name__}."
                )
            if not settings.boards:
                raise ValueError("RFSoCConfig.boards cannot be empty.")
            return settings

        if not self.address:
            raise ValueError(
                f"Missing {self.config!r} RFSoCConfig and no legacy address was set."
            )
        try:
            host, port = self.address.rsplit(":", maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                "Legacy RFSoC address must have the form 'host:port'."
            ) from exc

        return RFSoCConfig(
            boards=[BoardSettings(host=host, port=int(port))],
            ro_time_of_flight=self.cfg.ro_time_of_flight,
            soft_avgs=self.cfg.soft_avgs,
        )

    def _execute(
        self,
        configs: dict[str, Config],
        sequence: PulseSequence,
        sweepers: list[ParallelSweepers],
        opcode: rfsoc.OperationCode,
        options: ExecutionParameters,
    ) -> list[tuple[list, list]]:
        """Build one command per board and execute all commands concurrently."""
        settings = self._board_settings(configs)
        board_count = len(settings.boards)

        converted_sweepers = [
            [convert_units_sweeper(s, self.channels, configs) for s in parsweep]
            for parsweep in sweepers
        ]
        if len(sweepers) > 0:
            if opcode == rfsoc.OperationCode.EXECUTE_PULSE_SEQUENCE_RAW:
                raise RuntimeError("Sweep not permitted in RAW mode.")
            opcode = rfsoc.OperationCode.EXECUTE_SWEEPS

        if board_count > 1 and converted_sweepers:
            raise RuntimeError(
                "Internal error: hardware sweepers must be converted to software "
                "sweepers before a multi-board execution."
            )

        commands = []
        hosts = []
        ports = []

        for board, connection in enumerate(settings.boards):
            board_sequence = PulseSequence(
                [
                    (ch, element)
                    for ch, element in sequence
                    if _channel_board(ch, board_count) == board
                ]
            )
            board_channels = {
                ch: channel
                for ch, channel in self.channels.items()
                if _channel_board(ch, board_count) == board
            }

            qubits = [
                rfsoc.Qubit(
                    bias=getattr(configs[ch], "offset", 0.0),
                    dac=int(channel.path),
                )
                for ch, channel in board_channels.items()
                if isinstance(channel, DcChannel)
            ]

            start = (
                rfsoc.StartMode.START_IMMEDIATE
                if board_count == 1
                else (
                    rfsoc.StartMode.START_MASTER
                    if board == 0
                    else rfsoc.StartMode.START_SLAVE
                )
            )
            cfg = replace(
                self.cfg,
                ro_time_of_flight=settings.ro_time_of_flight,
                soft_avgs=settings.soft_avgs,
                start=start,
                delay=connection.delay,
                timeout=connection.timeout,
            )
            _update_cfg(cfg, options)

            # Keep dataclass objects here. execute_multiple -> execute ->
            # convert_commands owns serialization in the new qibosoq client.
            commands.append(
                {
                    "operation_code": opcode,
                    "cfg": cfg,
                    "sequence": convert(
                        board_sequence,
                        self.sampling_rate,
                        board_channels,
                        configs,
                    ),
                    "qubits": qubits,
                    "sweepers": [
                        convert(parsweep, board_sequence, board_channels)
                        for parsweep in converted_sweepers
                    ],
                }
            )
            hosts.append(connection.host)
            ports.append(connection.port)

        try:
            # execute_multiple staggers submissions. Arm all slaves first and
            # submit the master last so that its trigger cannot precede them.
            execution_order = [*range(1, board_count), 0] if board_count > 1 else [0]
            ordered_results = client.execute_multiple(
                [commands[board] for board in execution_order],
                [hosts[board] for board in execution_order],
                [ports[board] for board in execution_order],
                max_retries=settings.max_retries,
            )
            results_by_board: list[tuple[list, list] | None] = [None] * board_count
            for board, result in zip(execution_order, ordered_results):
                results_by_board[board] = result
            assert all(result is not None for result in results_by_board)
            return cast(list[tuple[list, list]], results_by_board)
        except RuntimeError as exc:
            cause = exc.__cause__ or exc
            if isinstance(cause, client.RuntimeLoopError) or (
                "exception in readout loop" in str(cause)
            ):
                log.warning(
                    "%s %s",
                    "Exception in readout loop after qibosoq client retries.",
                    "You may want to increase the relaxation time.",
                )
            buffer_overflow = r"buffer length must be \d+ samples or less"
            if isinstance(cause, client.BufferLengthError) or (
                re.search(buffer_overflow, str(cause)) is not None
            ):
                log.warning("Buffer full! Use shorter pulses.")
            raise


def _channel_board(channel: ChannelId, board_count: int) -> int:
    """Extract the board index from a ``<board>_...`` channel identifier."""
    match = re.match(r"^(\d+)_", str(channel))
    if match is None:
        if board_count == 1:
            return 0
        raise ValueError(
            f"Channel {channel!r} has no board prefix. Multi-board channel "
            "identifiers must start with '<board>_', for example "
            "'0_0/drive'."
        )

    board = int(match.group(1))
    if board >= board_count:
        raise ValueError(
            f"Channel {channel!r} selects board {board}, but only "
            f"{board_count} boards are configured."
        )
    return board


def _validate_board_channels(
    channels: dict[ChannelId, Channel], board_count: int
) -> None:
    """Validate all channel prefixes before starting any remote execution."""
    for channel in channels:
        _channel_board(channel, board_count)


def _validate_input_command(
    sequence: PulseSequence,
    options: ExecutionParameters,
    sweepers: list[ParallelSweepers],
):
    """Check if sequence and execution_parameters are supported."""
    if options.acquisition_type is AcquisitionType.RAW:
        if len(sweepers) > 0:
            raise NotImplementedError(
                "Raw data acquisition is not compatible with sweepers"
            )
        if len(sequence.acquisitions) != 1:
            raise NotImplementedError(
                "Raw data acquisition is compatible only with a single readout"
            )
        if options.averaging_mode is not AveragingMode.CYCLIC:
            raise NotImplementedError("Raw data acquisition can only be averaged")
    if options.fast_reset:
        raise NotImplementedError("Fast reset is not supported")


def _update_cfg(cfg, options: ExecutionParameters):
    """Update rfsoc.Config object with new parameters."""
    if options.nshots is not None:
        cfg.reps = options.nshots
    if options.relaxation_time is not None:
        cfg.relaxation_time = options.relaxation_time * nano / micro


def _firmware_loops(
    sequence: PulseSequence,
    sweepers: list[ParallelSweepers],
    configs: dict[str, Config],
    channels: dict[ChannelId, Channel],
) -> int:
    """Check if a sweeper must be run with python loop or on hardware.

    To be run on qick internal loop a sweep must:
        * not be on the readout frequency
        * not be a duration sweeper
        * not be a duration_interpolated sweeper
        * only one pulse per channel supported
        * flux pulses are not compatible with sweepers

    Args:
        sequence (`qibolab.pulses.PulseSequence`). Pulse sequence to play.
        *sweepers (`qibosoq.abstract.Sweeper`): Sweeper objects.
    Returns:
        A boolean value true if the sweeper must be executed by python
        loop, false otherwise
    """
    return 0  # TODO for tprocv2
    if any(
        isinstance(p, Pulse) and isinstance(configs[ch], DcConfig) for ch, p in sequence
    ):
        return 0

    n = 0
    for parsweep in reversed(sweepers):
        if any(s.parameter is Parameter.duration_interpolated for s in parsweep):
            return n
        if any(s.parameter is Parameter.duration for s in parsweep):
            return n

        if any(
            s.parameter is Parameter.frequency
            and s.channels is not None
            and any(
                (isinstance(ch, AcquisitionChannel) or "probe" in ch)
                for ch in s.channels
            )
            for s in parsweep
        ):
            # if it's a sweep on the readout freq do a python sweep
            return n

        for s in parsweep:
            if s.channels is not None:
                this_channels = s.channels
            else:
                assert s.pulses is not None
                this_channels = [
                    ch for p in s.pulses for ch in sequence.pulse_channels(p.id)
                ]

            # If more than a pulse is on the sweeped channel this is a soft loop
            # What matters is the DAC number. Not the qibolab channel but the path
            for ch in this_channels:
                path = channels[ch].path
                pulses_in_path = []
                for p_ch, pulse in sequence:
                    # Type check so that ADC and DACs are not compared
                    if (
                        type(channels[p_ch]) is type(channels[ch])
                        and channels[p_ch].path == path
                    ):
                        pulses_in_path.append(pulse)

                if len(pulses_in_path) > 1:
                    return n

        # if not disallowed, increase the amount of firmware loops
        n += 1

    return n


def _update_sequence(sequence: PulseSequence, updates: dict) -> PulseSequence:
    """Apply sweep updates to base sequence."""
    return PulseSequence(
        [(ch, e.model_copy(update=updates.get(e.id, {}))) for ch, e in sequence]
    )


def _update_configs(configs: dict[str, Config], updates: dict) -> dict[str, Config]:
    """Apply sweep updates to base configs."""
    new_dict = {k: c.model_copy(update=updates.get(k, {})) for k, c in configs.items()}
    return new_dict


def _classify_shots(
    i: npt.NDArray, q: npt.NDArray, angle: float, threshold: float
) -> npt.NDArray:
    """Classify shots through linear separation."""
    rotated = np.cos(angle) * i - np.sin(angle) * q
    return np.heaviside(np.array(rotated) - threshold, 0)


def _merge_sweep_results(
    a: dict[PulseId, Result], b: dict[PulseId, Result]
) -> dict[PulseId, Result]:
    """Merge two results dictionaries, appending common keys."""
    return {
        key: np.append(a.get(key, []), b.get(key, [])) for key in a.keys() | b.keys()
    }


def _reshape_sweep_results(
    results: dict[PulseId, Result],
    sweepers: list[ParallelSweepers],
    execution_parameters: ExecutionParameters,
    firmware_loops: int,
) -> dict[PulseId, Result]:
    """Reshape result to correct Qibolab shape."""
    if execution_parameters.acquisition_type is AcquisitionType.RAW:
        return results

    shape = [len(sweeper[0].values) for sweeper in sweepers]  # pyright: ignore

    is_not_cyclic = execution_parameters.averaging_mode is not AveragingMode.CYCLIC
    is_discrim = execution_parameters.acquisition_type is AcquisitionType.DISCRIMINATION

    if is_not_cyclic:
        shape.append(getattr(execution_parameters, "nshots", 1))

    if not is_discrim:
        shape.append(2)  # I/Q last axis
    elif is_not_cyclic and firmware_loops != 0:
        shape = [shape[-1]] + shape[:-1]

    reshaped = {}
    for key, value in results.items():
        assert value.size == np.prod(shape), (
            f"Size mismatch: value.size={value.size}, expected {np.prod(shape)}, shape={shape}"
        )

        reshaped[key] = value.reshape(shape)
        if is_not_cyclic:
            if is_discrim and firmware_loops == 0:
                reshaped[key] = np.moveaxis(reshaped[key], 0, -1)
            elif is_discrim and firmware_loops == 1:
                reshaped[key] = np.moveaxis(reshaped[key], 0, -1)
                reshaped[key] = value.reshape([shape[0], shape[2], shape[1]])
            else:
                reshaped[key] = np.moveaxis(reshaped[key], 0, -2)

    return reshaped
