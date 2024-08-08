"""RFSoC FPGA driver."""

import re
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
import qibosoq.components.base as rfsoc
from qibo.config import log
from qibosoq import client

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.couplers import Coupler
from qibolab.instruments.abstract import Controller
from qibolab.pulses import PulseSequence
from qibolab.qubits import Qubit
from qibolab.sweeper import BIAS, Sweeper

from .convert import convert, convert_units_sweeper

HZ_TO_MHZ = 1e-6
NS_TO_US = 1e-3


@dataclass
class RFSoCPort:
    """Port object of the RFSoC."""

    name: int
    """DAC number."""
    offset: float = 0.0
    """Amplitude factor for biasing."""


class RFSoC(Controller):
    """Instrument object for controlling RFSoC FPGAs.

    The two way of executing pulses are with ``play`` (for arbitrary
    qibolab ``PulseSequence``) or with ``sweep`` that execute a
    ``PulseSequence`` object with one or more ``Sweeper``.

    Attributes:
        cfg (rfsoc.Config): Configuration dictionary required for pulse execution.
    """

    PortType = RFSoCPort

    def __init__(self, name: str, address: str, port: int, sampling_rate: float = 1.0):
        """Set server information and base configuration.

        Args:
            name (str): Name of the instrument instance.
            address (str): IP and port of the server (ex. 192.168.0.10)
            port (int): Port of the server (ex.6000)
        """
        super().__init__(name, address=address)
        self.host = address
        self.port = port
        self.cfg = rfsoc.Config()
        self._sampling_rate = sampling_rate

    @property
    def sampling_rate(self):
        return self._sampling_rate

    def connect(self):
        """Empty method to comply with Instrument interface."""

    def disconnect(self):
        """Empty method to comply with Instrument interface."""

    @staticmethod
    def _try_to_execute(server_commands, host, port):
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

    @staticmethod
    def convert_and_discriminate_samples(discriminated_shots, execution_parameters):
        if execution_parameters.averaging_mode is AveragingMode.CYCLIC:
            _, counts = np.unique(discriminated_shots, return_counts=True, axis=0)
            freqs = counts / discriminated_shots.shape[0]
            result = execution_parameters.results_type(freqs, discriminated_shots)
        else:
            result = execution_parameters.results_type(discriminated_shots)
        return result

    @staticmethod
    def validate_input_command(
        sequence: PulseSequence, execution_parameters: ExecutionParameters, sweep: bool
    ):
        """Check if sequence and execution_parameters are supported."""
        if execution_parameters.acquisition_type is AcquisitionType.RAW:
            if sweep:
                raise NotImplementedError(
                    "Raw data acquisition is not compatible with sweepers"
                )
            if len(sequence.probe_pulses) != 1:
                raise NotImplementedError(
                    "Raw data acquisition is compatible only with a single readout"
                )
            if execution_parameters.averaging_mode is not AveragingMode.CYCLIC:
                raise NotImplementedError("Raw data acquisition can only be averaged")
        if execution_parameters.fast_reset:
            raise NotImplementedError("Fast reset is not supported")

    @staticmethod
    def merge_sweep_results(
        dict_a: dict[str, npt.NDArray],
        dict_b: dict[str, npt.NDArray],
    ) -> dict[str, npt.NDArray]:
        """Merge two dictionary mapping pulse serial to Results object.

        If dict_b has a key (serial) that dict_a does not have, simply add it,
        otherwise sum the two results

        Args:
            dict_a (dict): dict mapping ro pulses serial to qibolab res objects
            dict_b (dict): dict mapping ro pulses serial to qibolab res objects
        Returns:
            A dict mapping the readout pulses serial to qibolab results objects
        """
        for serial in dict_b:
            if serial in dict_a:
                dict_a[serial] = np.append(dict_a[serial], dict_b[serial])
            else:
                dict_a[serial] = dict_b[serial]
        return dict_a

    @staticmethod
    def reshape_sweep_results(
        results, sweepers, execution_parameters: ExecutionParameters
    ):
        # TODO: pay attention: the following will not work in raw waveform acquisition
        # modes, in which case the number of samples taken should be passed as an
        # explicit parameter
        shape = execution_parameters.results_shape(sweepers)
        return {key: value.reshape(shape) for key, value in results.items()}

    def _execute_pulse_sequence(
        self,
        sequence: PulseSequence,
        qubits: dict[int, Qubit],
        opcode: rfsoc.OperationCode,
    ) -> tuple[list, list]:
        """Prepare the commands dictionary to send to the qibosoq server.

        Args:
            sequence (`qibolab.pulses.PulseSequence`): arbitrary PulseSequence object to execute
            qubits: list of qubits (`qibolab.platforms.abstract.Qubit`) of the platform in the form of a dictionary
            opcode: can be `rfsoc.OperationCode.EXECUTE_PULSE_SEQUENCE` or `rfsoc.OperationCode.EXECUTE_PULSE_SEQUENCE_RAW`
        Returns:
            Lists of I and Q value measured
        """
        server_commands = {
            "operation_code": opcode,
            "cfg": asdict(self.cfg),
            "sequence": convert(sequence, qubits, self.sampling_rate),
            "qubits": [asdict(convert(qubits[idx])) for idx in qubits],
        }
        return self._try_to_execute(server_commands, self.host, self.port)

    def _execute_sweeps(
        self,
        sequence: PulseSequence,
        qubits: dict[int, Qubit],
        sweepers: list[rfsoc.Sweeper],
    ) -> tuple[list, list]:
        """Prepare the commands dictionary to send to the qibosoq server.

        Args:
            sequence (`qibolab.pulses.PulseSequence`): arbitrary PulseSequence object to execute
            qubits: list of qubits (`qibolab.platforms.abstract.Qubit`) of the platform in the form of a dictionary
            sweepers: list of `qibosoq.abstract.Sweeper` objects
        Returns:
            Lists of I and Q value measured
        """
        converted_sweepers = [
            convert_units_sweeper(sweeper, sequence, qubits) for sweeper in sweepers
        ]
        server_commands = {
            "operation_code": rfsoc.OperationCode.EXECUTE_SWEEPS,
            "cfg": asdict(self.cfg),
            "sequence": convert(sequence, qubits, self.sampling_rate),
            "qubits": [asdict(convert(qubits[idx])) for idx in qubits],
            "sweepers": [sweeper.serialized for sweeper in converted_sweepers],
        }
        return self._try_to_execute(server_commands, self.host, self.port)

    def play(
        self,
        qubits: dict[int, Qubit],
        couplers: dict[int, Coupler],
        sequence: PulseSequence,
        execution_parameters: ExecutionParameters,
    ) -> dict[str, npt.NDArray]:
        """Execute the sequence of instructions and retrieves readout results.

        Each readout pulse generates a separate acquisition.
        The relaxation_time and the number of shots have default values.

        Args:
            qubits (dict): List of `qibolab.platforms.utils.Qubit` objects
                           passed from the platform.
            execution_parameters (`qibolab.ExecutionParameters`): Parameters (nshots,
                                                        relaxation_time,
                                                        fast_reset,
                                                        acquisition_type,
                                                        averaging_mode)
            sequence (`qibolab.pulses.PulseSequence`): Pulse sequence to play.
        Returns:
            A dictionary mapping the readout pulses serial and respective qubits to
            qibolab results objects
        """
        if couplers != {}:
            raise NotImplementedError(
                "The RFSoC driver currently does not support couplers."
            )

        self.validate_input_command(sequence, execution_parameters, sweep=False)
        self.update_cfg(execution_parameters)

        if execution_parameters.acquisition_type is AcquisitionType.DISCRIMINATION:
            self.cfg.average = False
        else:
            self.cfg.average = (
                execution_parameters.averaging_mode is AveragingMode.CYCLIC
            )

        if execution_parameters.acquisition_type is AcquisitionType.RAW:
            opcode = rfsoc.OperationCode.EXECUTE_PULSE_SEQUENCE_RAW
        else:
            opcode = rfsoc.OperationCode.EXECUTE_PULSE_SEQUENCE
        toti, totq = self._execute_pulse_sequence(sequence, qubits, opcode)

        results = {}
        probed_qubits = np.unique([p.qubit for p in sequence.probe_pulses])

        for j, qubit in enumerate(probed_qubits):
            for i, ro_pulse in enumerate(sequence.probe_pulses.get_qubit_pulses(qubit)):
                i_pulse = np.array(toti[j][i])
                q_pulse = np.array(totq[j][i])

                if (
                    execution_parameters.acquisition_type
                    is AcquisitionType.DISCRIMINATION
                ):
                    discriminated_shots = self.classify_shots(
                        i_pulse, q_pulse, qubits[ro_pulse.qubit]
                    )
                    result = self.convert_and_discriminate_samples(
                        discriminated_shots, execution_parameters
                    )
                else:
                    result = execution_parameters.results_type(i_pulse + 1j * q_pulse)
                results[ro_pulse.qubit] = results[ro_pulse.id] = result

        return results

    def update_cfg(self, execution_parameters: ExecutionParameters):
        """Update rfsoc.Config object with new parameters."""
        if execution_parameters.nshots is not None:
            self.cfg.reps = execution_parameters.nshots
        if execution_parameters.relaxation_time is not None:
            self.cfg.relaxation_time = execution_parameters.relaxation_time * NS_TO_US

    def classify_shots(
        self,
        i_values: npt.NDArray[np.float64],
        q_values: npt.NDArray[np.float64],
        qubit: Qubit,
    ) -> npt.NDArray[np.float64]:
        """Classify IQ values using qubit threshold and rotation_angle if
        available in runcard."""
        if qubit.iq_angle is None or qubit.threshold is None:
            raise ValueError("Classification parameters were not provided")
        angle = qubit.iq_angle
        threshold = qubit.threshold

        rotated = np.cos(angle) * np.array(i_values) - np.sin(angle) * np.array(
            q_values
        )
        shots = np.heaviside(np.array(rotated) - threshold, 0)
        if isinstance(shots, float):
            return np.array([shots])
        return shots

    def play_sequence_in_sweep_recursion(
        self,
        qubits: dict[int, Qubit],
        couplers: dict[int, Coupler],
        sequence: PulseSequence,
        or_sequence: PulseSequence,
        execution_parameters: ExecutionParameters,
    ) -> dict[str, npt.NDArray]:
        """Last recursion layer, if no sweeps are present.

        After playing the sequence, the resulting dictionary keys need
        to be converted to the correct values. Even indexes correspond
        to qubit number and are not changed. Odd indexes correspond to
        readout pulses serials and are convert to match the original
        sequence (of the sweep) and not the one just executed.
        """
        res = self.play(qubits, couplers, sequence, execution_parameters)
        newres = {}
        serials = [pulse.id for pulse in or_sequence.probe_pulses]
        for idx, key in enumerate(res):
            if idx % 2 == 1:
                newres[serials[idx // 2]] = res[key]
            else:
                newres[key] = res[key]

        return newres

    def recursive_python_sweep(
        self,
        qubits: dict[int, Qubit],
        couplers: dict[int, Coupler],
        sequence: PulseSequence,
        or_sequence: PulseSequence,
        *sweepers: rfsoc.Sweeper,
        execution_parameters: ExecutionParameters,
    ) -> dict[str, npt.NDArray]:
        """Execute a sweep of an arbitrary number of Sweepers via recursion.

        Args:
            qubits (list): List of `qibolab.platforms.utils.Qubit` objects
                    passed from the platform.
            sequence (`qibolab.pulses.PulseSequence`): Pulse sequence to play.
                    This object is a deep copy of the original
                    sequence and gets modified.
            or_sequence (`qibolab.pulses.PulseSequence`): Reference to original
                    sequence to not modify.
            *sweepers (`qibolab.Sweeper`): Sweeper objects.
            execution_parameters (`qibolab.ExecutionParameters`): Parameters (nshots,
                                                        relaxation_time,
                                                        fast_reset,
                                                        acquisition_type,
                                                        averaging_mode)
        Returns:
            A dictionary mapping the readout pulses serial and respective qubits to
            results objects
        """
        # If there are no sweepers run ExecutePulseSequence acquisition.
        # Last layer for recursion.

        if len(sweepers) == 0:
            return self.play_sequence_in_sweep_recursion(
                qubits, couplers, sequence, or_sequence, execution_parameters
            )

        if not self.get_if_python_sweep(sequence, *sweepers):
            toti, totq = self._execute_sweeps(sequence, qubits, sweepers)
            res = self.convert_sweep_results(
                or_sequence, qubits, toti, totq, execution_parameters
            )
            return res

        sweeper = sweepers[0]
        values = []
        for idx, _ in enumerate(sweeper.indexes):
            val = np.linspace(sweeper.starts[idx], sweeper.stops[idx], sweeper.expts)
            if sweeper.parameters[idx] in rfsoc.Parameter.variants(
                {"duration", "delay"}
            ):
                val = val.astype(int)
            values.append(val)

        results: dict[str, npt.NDArray] = {}
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

            res = self.recursive_python_sweep(
                qubits,
                couplers,
                sequence,
                or_sequence,
                *sweepers[1:],
                execution_parameters=execution_parameters,
            )
            results = self.merge_sweep_results(results, res)
        return results

    def get_if_python_sweep(
        self, sequence: PulseSequence, *sweepers: rfsoc.Sweeper
    ) -> bool:
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
        # FIXME: since pulse types have been deprecated, now channel types should be
        # used instead. In the following, the code is relying on a non-standard channels
        # naming convention, and thus fragile (or just broken)
        # instead, the channel object should be retrieved from the platform
        # configuration, and its type should be inspected
        if any("flux" in ch for ch in sequence):
            return True
        for sweeper in sweepers:
            if all(
                parameter is rfsoc.Parameter.BIAS for parameter in sweeper.parameters
            ):
                continue
            if all(
                parameter is rfsoc.Parameter.DELAY for parameter in sweeper.parameters
            ):
                continue
            if any(
                parameter is rfsoc.Parameter.DURATION
                for parameter in sweeper.parameters
            ):
                return True

            for sweep_idx, parameter in enumerate(sweeper.parameters):
                is_freq = parameter is rfsoc.Parameter.FREQUENCY
                # FIXME: the sequence can not even be indexed like this any longer
                # is_ro = sequence[sweeper.indexes[sweep_idx]].type == PulseType.READOUT
                is_ro = False
                # if it's a sweep on the readout freq do a python sweep
                if is_freq and is_ro:
                    return True

            for idx in sweeper.indexes:
                sweep_pulse = sequence[idx]
                channel = sweep_pulse.channel
                ch_pulses = sequence.get_channel_pulses(channel)
                if len(ch_pulses) > 1:
                    return True
        # if all passed, do a firmware sweep
        return False

    def convert_sweep_results(
        self,
        original_ro: PulseSequence,
        qubits: dict[int, Qubit],
        toti: list[list[list[float]]],
        totq: list[list[list[float]]],
        execution_parameters: ExecutionParameters,
    ) -> dict[str, npt.NDArray]:
        """Convert sweep res to qibolab dict res.

        Args:
            original_ro (`qibolab.pulses.PulseSequence`): Original PulseSequence
            qubits (list): List of `qibolab.platforms.utils.Qubit` objects
                 passed from the platform.
            toti (list): i values
            totq (list): q values
            results_type: qibolab results object
            execution_parameters (`qibolab.ExecutionParameters`): Parameters (nshots,
                                                        relaxation_time,
                                                        fast_reset,
                                                        acquisition_type,
                                                        averaging_mode)
        Returns:
            A dict mapping the readout pulses serial to qibolab results objects
        """
        results = {}

        adcs = np.unique([qubits[p.qubit].feedback.port.name for p in original_ro])
        for k, k_val in enumerate(adcs):
            adc_ro = [
                pulse
                for pulse in original_ro
                if qubits[pulse.qubit].feedback.port.name == k_val
            ]
            for i, ro_pulse in enumerate(adc_ro):
                i_vals = np.array(toti[k][i])
                q_vals = np.array(totq[k][i])

                if not self.cfg.average:
                    i_vals = np.reshape(i_vals, (self.cfg.reps, *i_vals.shape[:-1]))
                    q_vals = np.reshape(q_vals, (self.cfg.reps, *q_vals.shape[:-1]))

                if (
                    execution_parameters.acquisition_type
                    is AcquisitionType.DISCRIMINATION
                ):
                    qubit = qubits[ro_pulse.qubit]
                    discriminated_shots = self.classify_shots(i_vals, q_vals, qubit)
                    result = self.convert_and_discriminate_samples(
                        discriminated_shots, execution_parameters
                    )

                else:
                    result = execution_parameters.results_type(i_vals + 1j * q_vals)

                results[ro_pulse.qubit] = results[ro_pulse.id] = result
        return results

    def sweep(
        self,
        qubits: dict[int, Qubit],
        couplers: dict[int, Coupler],
        sequence: PulseSequence,
        execution_parameters: ExecutionParameters,
        *sweepers: Sweeper,
    ) -> dict[str, npt.NDArray]:
        """Execute the sweep and retrieves the readout results.

        Each readout pulse generates a separate acquisition.
        The relaxation_time and the number of shots have default values.

        Args:
            qubits (list): List of `qibolab.platforms.utils.Qubit` objects
                           passed from the platform.
            execution_parameters (`qibolab.ExecutionParameters`): Parameters (nshots,
                                                        relaxation_time,
                                                        fast_reset,
                                                        acquisition_type,
                                                        averaging_mode)
            sequence (`qibolab.pulses.PulseSequence`). Pulse sequence to play.
            *sweepers (`qibolab.Sweeper`): Sweeper objects.
        Returns:
            A dictionary mapping the readout pulses serial and respective qubits to
            results objects
        """
        if couplers != {}:
            raise NotImplementedError(
                "The RFSoC driver currently does not support couplers."
            )

        self.validate_input_command(sequence, execution_parameters, sweep=True)
        self.update_cfg(execution_parameters)

        if execution_parameters.acquisition_type is AcquisitionType.DISCRIMINATION:
            self.cfg.average = False
        else:
            self.cfg.average = (
                execution_parameters.averaging_mode is AveragingMode.CYCLIC
            )

        rfsoc_sweepers = [convert(sweep, sequence, qubits) for sweep in sweepers]

        sweepsequence = sequence.copy()

        bias_change = any(sweep.parameter is BIAS for sweep in sweepers)
        if bias_change:
            initial_biases = [
                qubits[idx].flux.offset if qubits[idx].flux is not None else None
                for idx in qubits
            ]

        results = self.recursive_python_sweep(
            qubits,
            couplers,
            sweepsequence,
            sequence.probe_pulses,
            *rfsoc_sweepers,
            execution_parameters=execution_parameters,
        )

        if bias_change:
            for idx, qubit in enumerate(qubits.values()):
                if qubit.flux is not None:
                    qubit.flux.offset = initial_biases[idx]

        return self.reshape_sweep_results(results, sweepers, execution_parameters)
