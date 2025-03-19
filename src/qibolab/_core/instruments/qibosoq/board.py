"""QICK-Qibosoq interface."""

import re
from dataclasses import asdict

import numpy as np
import numpy.typing as npt
import qibosoq.components.base as rfsoc
from pydantic import Field
from qibo.config import log
from qibosoq import client
from scipy.constants import micro, nano

from qibolab._core.components.channels import AcquisitionChannel
from qibolab._core.components.configs import Config, DcConfig
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.pulses import Pulse
from qibolab._core.qubits import Qubit
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers, Parameter

from .convert import convert, convert_units_sweeper


class RFSoC(Controller):
    """Instrument controlling RFSoC FPGAs."""

    sampling_rate: float
    cfg: rfsoc.Config = Field(default_factory=rfsoc.Config)
    """Configuration dictionary required for pulse execution."""

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
    ) -> dict[int, Result]:
        """Play a pulse sequence and retrieve feedback."""
        results = {}

        for seq in sequences:
            _validate_input_command(seq, options, sweepers)
            _update_cfg(self.cfg, options)

            fw = _firmware_loops(seq, sweepers, configs)
            res = self._sweep(configs, seq, sweepers, len(sweepers) - fw, options)
            results |= _reshape_sweep_results(res, sweepers, options)

        return results

    def _sweep(
        self,
        configs: dict[str, Config],
        sequence: PulseSequence,
        sweepers: list[ParallelSweepers],
        software: int,
        options: ExecutionParameters,
    ) -> dict[int, Result]:
        """Execute a sweep of an arbitrary number of Sweepers via recursion."""
        # If there are no sweepers run ExecutePulseSequence acquisition.
        # Last layer for recursion.

        if len(sweepers) == 0:
            return self._play_sequence_in_sweep_recursion(
                qubits, couplers, sequence, or_sequence, execution_parameters
            )

        if software == 0:
            toti, totq = self._execute_sweeps(sequence, qubits, sweepers)
            res = self._convert_sweep_results(
                or_sequence, qubits, toti, totq, execution_parameters
            )
            return res

        sweeper = convert(sweepers[0][0], sequence, configs)
        values = []
        for idx, _ in enumerate(sweeper.indexes):
            val = np.linspace(sweeper.starts[idx], sweeper.stops[idx], sweeper.expts)
            if sweeper.parameters[idx] in rfsoc.Parameter.variants(
                {"duration", "delay"}
            ):
                val = val.astype(int)
            values.append(val)

        results = {}
        for idx in range(sweeper.expts):
            # update values
            for jdx, kdx in enumerate(sweeper.indexes):
                sweeper_parameter = sweeper.parameters[jdx]
                if sweeper_parameter is rfsoc.Parameter.BIAS:
                    qubits[list(qubits)[kdx]].flux.offset = values[jdx][idx]
                elif sweeper_parameter in rfsoc.Parameter.variants(
                    {
                        "amplitude",
                        "frequency",
                        "relative_phase",
                        "duration",
                    }
                ):
                    setattr(
                        sequence[kdx], sweeper_parameter.name.lower(), values[jdx][idx]
                    )
                    if sweeper_parameter is rfsoc.Parameter.DURATION:
                        for pulse_idx in range(
                            kdx + 1,
                            len(sequence.get_qubit_pulses(sequence[kdx].qubit)),
                        ):
                            # TODO: this is a patch and works just for simple experiments
                            sequence[pulse_idx].start = sequence[pulse_idx - 1].finish
                elif sweeper_parameter is rfsoc.Parameter.DELAY:
                    sequence[kdx].start_delay = values[jdx][idx]

            res = self._sweep(configs, sequence, sweepers[1:], software - 1, options)
            results = _merge_sweep_results(results, res)
        return results

    def _play_sequence_in_sweep_recursion(
        self,
        qubits: dict[int, Qubit],
        sequence: PulseSequence,
        or_sequence: PulseSequence,
        execution_parameters: ExecutionParameters,
    ) -> dict[int, Result]:
        """Last recursion layer, if no sweeps are present.

        After playing the sequence, the resulting dictionary keys need
        to be converted to the correct values. Even indexes correspond
        to qubit number and are not changed. Odd indexes correspond to
        readout pulses serials and are convert to match the original
        sequence (of the sweep) and not the one just executed.
        """
        res = self._play(qubits, couplers, sequence, execution_parameters)
        newres = {}
        serials = [pulse.serial for pulse in or_sequence.ro_pulses]
        for idx, key in enumerate(res):
            if idx % 2 == 1:
                newres[serials[idx // 2]] = res[key]
            else:
                newres[key] = res[key]

        return newres

    def _execute(
        self,
        sequence: PulseSequence,
        qubits: dict[int, Qubit],
        sweepers: list[rfsoc.Sweeper],
        opcode: rfsoc.OperationCode,
    ) -> tuple[list, list]:
        """Prepare the commands dictionary to send to the qibosoq server.

        Returns lists of I and Q value measured.
        """
        host, port_ = self.address.split(":")
        port = int(port_)
        converted_sweepers = [
            convert_units_sweeper(sweeper, sequence, qubits) for sweeper in sweepers
        ]
        server_commands = {
            "operation_code": opcode,
            "cfg": asdict(self.cfg),
            "sequence": convert(sequence, qubits, self.sampling_rate),
            "qubits": [asdict(convert(qubits[idx])) for idx in qubits],
            "sweepers": [sweeper.serialized for sweeper in converted_sweepers],
        }
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

    def _play(
        self,
        configs: dict[str, Config],
        sequence: PulseSequence,
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[int, Result]:
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
        toti, totq = self._execute(sequence, configs, opcode, sweepers)

        probed_qubits = np.unique([p.qubit for p in sequence.ro_pulses])

        for j, qubit in enumerate(probed_qubits):
            for i, ro_pulse in enumerate(sequence.ro_pulses):
                i_pulse = np.array(toti[j][i])
                q_pulse = np.array(totq[j][i])

                if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                    result = self._classify_shots(i_pulse, q_pulse, angle, threshold)
                else:
                    result = np.stack([i_pulse, q_pulse], axis=-1)
                results[ro_pulse.serial] = result

        return results

    def _classify_shots(
        self,
        i_values: npt.NDArray[np.float64],
        q_values: npt.NDArray[np.float64],
        angle: float,
        threshold: float,
    ) -> npt.NDArray[np.float64]:
        rotated = (
            np.asarray([np.cos(angle), -np.sin(angle)])
            * np.stack([i_values, q_values]).T
        ).T
        return np.heaviside(np.array(rotated) - threshold, 0)

    def _convert_sweep_results(
        self,
        sequence: PulseSequence,
        configs: dict[str, Config],
        toti: list[list[list[float]]],
        totq: list[list[list[float]]],
        options: ExecutionParameters,
    ) -> dict[str, Result]:
        """Convert sweep res to qibolab dict res.

        `toti` are the I values downloaded, and `totq` Q ones.
        """
        results = {}

        adcs = np.unique([configs[ch].path.port for ch in sequence.channels])
        for k, k_val in enumerate(adcs):
            adc_ro = [
                pulse
                for _, pulse in sequence
                if qubits[pulse.qubit].feedback.port.name == k_val
            ]
            for i, ro_pulse in enumerate(adc_ro):
                i_vals = np.array(toti[k][i])
                q_vals = np.array(totq[k][i])

                if not self.cfg.average:
                    i_vals = np.reshape(i_vals, (self.cfg.reps, *i_vals.shape[:-1]))
                    q_vals = np.reshape(q_vals, (self.cfg.reps, *q_vals.shape[:-1]))

                if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                    result = self._classify_shots(i_vals, q_vals, angle, threshold)
                else:
                    result = np.stack([i_vals, q_vals], axis=-1)

                results[ro_pulse.id] = result
        return results


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
        if len(sequence.ro_pulses) != 1:
            raise NotImplementedError(
                "Raw data acquisition is compatible only with a single readout"
            )
        if options.averaging_mode is not AveragingMode.CYCLIC:
            raise NotImplementedError("Raw data acquisition can only be averaged")
    if options.fast_reset:
        raise NotImplementedError("Fast reset is not supported")


def _merge_sweep_results(
    a: dict[int, Result], b: dict[int, Result]
) -> dict[int, Result]:
    """Merge two results dictionaries, appending common keys."""
    return {
        key: np.append(a.get(key, []), b.get(key, [])) for key in a.keys() | b.keys()
    }


def _reshape_sweep_results(results, sweepers, execution_parameters):
    shape = [len(sweeper.values) for sweeper in sweepers]
    if execution_parameters.averaging_mode is not AveragingMode.CYCLIC:
        shape.insert(0, execution_parameters.nshots)

    return {key: value.reshape(shape) for key, value in results.items()}


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
            and any(isinstance(ch, AcquisitionChannel) for ch in s.channels)
            for s in parsweep
        ):
            # if it's a sweep on the readout freq do a python sweep
            return n

        for s in parsweep:
            channels = (
                s.channels
                if s.channels is not None
                else [ch for p in s.pulses for ch in sequence.pulse_channels(p.id)]
            )

            if any(sequence.channel(ch) for ch in channels) > 1:
                return n

        # if not disallowed, increase the amount of firmware loops
        n += 1

    return n
