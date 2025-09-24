"""QICK-Qibosoq interface."""

import re
from collections import defaultdict
from dataclasses import asdict
from typing import cast

import numpy as np
import numpy.typing as npt
import qibosoq.components.base as rfsoc
from pydantic import Field
from qibo.config import log
from qibosoq import client
from scipy.constants import micro, nano

from qibolab._core.components.channels import AcquisitionChannel, DcChannel
from qibolab._core.components.configs import AcquisitionConfig, Config, DcConfig
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.pulses import Pulse
from qibolab._core.pulses.pulse import PulseId
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers, Parameter

from .convert import convert, convert_units_sweeper

__all__ = ["RFSoC"]


class RFSoC(Controller):
    """Instrument controlling RFSoC FPGAs."""

    bounds: str = "rfsoc/bounds"
    _sampling_rate: float = 10e9
    cfg: rfsoc.Config = Field(default_factory=rfsoc.Config)
    """Configuration dictionary required for pulse execution."""

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
            _update_cfg(self.cfg, options)

            fw = _firmware_loops(seq, sweepers, configs)
            res = self._sweep(configs, seq, sweepers, len(sweepers) - fw, options, {})
            results |= _reshape_sweep_results(res, sweepers, options)

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
        results = {}

        # TODO: why not averaging for discrimination?
        self.cfg.average = (
            options.acquisition_type is not AcquisitionType.DISCRIMINATION
            and options.averaging_mode is AveragingMode.CYCLIC
        )
        opcode = (
            rfsoc.OperationCode.EXECUTE_PULSE_SEQUENCE_RAW
            if options.acquisition_type is AcquisitionType.RAW
            else rfsoc.OperationCode.EXECUTE_PULSE_SEQUENCE
        )
        toti, totq = self._execute(
            _update_configs(configs, updates),
            _update_sequence(sequence, updates),
            sweepers,
            opcode,
        )

        acq_chs = np.unique([acq[0] for acq in sequence.acquisitions])

        for idx, this_ch in enumerate(acq_chs):
            this_ch_acq = [
                (ch, acq) for ch, acq in sequence.acquisitions if ch == this_ch
            ]
            for i, q, (ch, acq) in zip(toti[idx], totq[idx], this_ch_acq):
                if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                    config = cast(AcquisitionConfig, configs[ch])
                    angle, threshold = config.iq_angle, config.threshold
                    assert angle is not None
                    assert threshold is not None
                    result = _classify_shots(np.array(i), np.array(q), angle, threshold)
                else:
                    result = np.stack([i, q], axis=-1)
                results[acq.id] = result

        return results

    def _execute(
        self,
        configs: dict[str, Config],
        sequence: PulseSequence,
        sweepers: list[ParallelSweepers],
        opcode: rfsoc.OperationCode,
    ) -> tuple[list, list]:
        """Prepare the commands dictionary to send to the qibosoq server.

        Returns lists of I and Q value measured.
        """
        converted_sweepers = [
            [convert_units_sweeper(s, self.channels, configs) for s in parsweep]
            for parsweep in sweepers
        ]
        if len(sweepers) > 0:
            if opcode == rfsoc.OperationCode.EXECUTE_PULSE_SEQUENCE_RAW:
                raise RuntimeError("Sweep not permitted in RAW mode.")
            opcode = rfsoc.OperationCode.EXECUTE_SWEEPS

        qubits = []
        for ch in self.channels:
            if isinstance(self.channels[ch], DcChannel):
                qubits.append(
                    rfsoc.Qubit(
                        bias=getattr(configs[ch], "offset", 0.0),
                        dac=int(self.channels[ch].path),
                    )
                )

        server_commands = {
            "operation_code": opcode,
            "cfg": asdict(self.cfg),
            "sequence": convert(sequence, self.sampling_rate, self.channels, configs),
            "qubits": [asdict(q) for q in qubits],
            "sweepers": [
                convert(parsweep, sequence, self.channels).serialized
                for parsweep in converted_sweepers
            ],
        }
        host, port_ = self.address.split(":")
        port = int(port_)

        try:
            return client.connect(server_commands, host, port)
        except RuntimeError as e:
            if "exception in readout loop" in str(e):
                log.warning(
                    "%s %s",
                    "Exception in readout loop. Attempting again",
                    "You may want to increase the relaxation time.",
                )
                return client.connect(server_commands, host, port)
            buffer_overflow = r"buffer length must be \d+ samples or less"
            if re.search(buffer_overflow, str(e)) is not None:
                log.warning("Buffer full! Use shorter pulses.")
            raise e


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
) -> int:
    """Check if a sweeper must be run with python loop or on hardware.

    To be run on qick internal loop a sweep must:
        * not be on the readout frequency
        * not be a duration sweeper
        * only one pulse per channel supported
        * flux pulses are not compatible with sweepers

    Args:
        sequence (`qibolab.pulses.PulseSequence`). Pulse sequence to play.
        *sweepers (`qibosoq.abstract.Sweeper`): Sweeper objects.
    Returns:
        A boolean value true if the sweeper must be executed by python
        loop, false otherwise
    """
    if any(
        isinstance(p, Pulse) and isinstance(configs[ch], DcConfig) for ch, p in sequence
    ):
        return 0

    n = 0
    for parsweep in reversed(sweepers):
        if any(
            s.parameter is Parameter.duration
            and s.pulses is not None
            and any(isinstance(p, Pulse) for p in s.pulses)
            for s in parsweep
        ):
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
                channels = s.channels
            else:
                assert s.pulses is not None
                channels = [
                    ch for p in s.pulses for ch in sequence.pulse_channels(p.id)
                ]

            if any(sequence.channel(ch) for ch in channels) > 1:
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
    return {k: c.model_copy(update=updates.get(k, {})) for k, c in configs.items()}


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
) -> dict[PulseId, Result]:
    """Reshape result to correct Qibolab shape."""
    if execution_parameters.acquisition_type is AcquisitionType.RAW:
        return results

    shape = [len(sweeper[0].values) for sweeper in sweepers]  # pyright: ignore

    if execution_parameters.averaging_mode is not AveragingMode.CYCLIC:
        shape.insert(0, getattr(execution_parameters, "nshots", 1))
    if execution_parameters.acquisition_type is not AcquisitionType.DISCRIMINATION:
        shape.append(2)  # I/Q last axis

    reshaped = {}
    for key, value in results.items():
        assert value.size == np.prod(shape), (
            f"Size mismatch: value.size={value.size}, expected {np.prod(shape)}, shape={shape}"
        )
        reshaped[key] = value.reshape(shape)

    return reshaped
