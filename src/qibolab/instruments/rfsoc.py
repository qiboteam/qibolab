"""RFSoC FPGA driver.

This driver needs a Qibosoq server installed and running
Tested on the following FPGA:
    - RFSoC 4x2
    - ZCU111
"""
import copy
import json
import socket
from dataclasses import asdict, dataclass
from typing import Union

import numpy as np
import qibosoq.components as rfsoc

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.instruments.abstract import Controller
from qibolab.instruments.port import Port
from qibolab.platform import Qubit
from qibolab.pulses import Pulse, PulseSequence, PulseShape, PulseType
from qibolab.result import IntegratedResults, SampleResults
from qibolab.sweeper import BIAS, DURATION, START, Parameter, Sweeper

HZ_TO_MHZ = 1e-6
NS_TO_US = 1e-3


def convert_qubit(qubit: Qubit) -> rfsoc.Qubit:
    """Convert `qibolab.platforms.abstract.Qubit` to `qibosoq.abstract.Qubit`."""
    if qubit.flux:
        return rfsoc.Qubit(qubit.flux.offset, qubit.flux.port.name)
    return rfsoc.Qubit(0.0, None)


def replace_pulse_shape(rfsoc_pulse: rfsoc.Pulse, shape: PulseShape) -> rfsoc.Pulse:
    """Set pulse shape parameters in rfsoc pulse object."""
    new = copy.copy(rfsoc_pulse)
    shape_name = new.shape = shape.name.lower()
    if shape_name in {"gaussian", "drag"}:
        new.rel_sigma = shape.rel_sigma
        if shape_name == "drag":
            new.beta = shape.beta
    return new


def pulse_lo_frequency(pulse: Pulse, qubits: dict[int, Qubit]) -> int:
    """Return local_oscillator frequency (HZ) of a pulse."""
    pulse_type = pulse.type.name.lower()
    try:
        lo_frequency = getattr(qubits[pulse.qubit], pulse_type).local_oscillator._frequency
    except AttributeError:
        lo_frequency = 0
    return lo_frequency


def convert_pulse_sequence(sequence: PulseSequence, qubits: dict[int, Qubit]) -> list[rfsoc.Pulse]:
    """Convert PulseSequence to list of rfosc pulses with relative time."""

    abs_time = 0
    list_sequence = []
    for pulse in sequence:
        abs_start = pulse.start * NS_TO_US
        start_delay = abs_start - abs_time
        pulse_dict = asdict(convert_pulse(pulse, qubits, start_delay))
        list_sequence.append(pulse_dict)

        abs_time += start_delay
    return list_sequence


def convert_pulse(pulse: Pulse, qubits: dict[int, Qubit], start_delay: float) -> rfsoc.Pulse:
    """Convert `qibolab.pulses.pulse` to `qibosoq.abstract.Pulse`."""
    pulse_type = pulse.type.name.lower()
    dac = getattr(qubits[pulse.qubit], pulse_type).port.name
    adc = qubits[pulse.qubit].feedback.port.name if pulse_type == "readout" else None
    lo_frequency = pulse_lo_frequency(pulse, qubits)

    rfsoc_pulse = rfsoc.Pulse(
        frequency=(pulse.frequency - lo_frequency) * HZ_TO_MHZ,
        amplitude=pulse.amplitude,
        relative_phase=np.degrees(pulse.relative_phase),
        start_delay=start_delay,
        duration=pulse.duration * NS_TO_US,
        dac=dac,
        adc=adc,
        shape=None,
        name=pulse.serial,
        type=pulse_type,
    )
    return replace_pulse_shape(rfsoc_pulse, pulse.shape)


def convert_units_sweeper(sweeper: rfsoc.Sweeper, sequence: PulseSequence, qubits: dict[int, Qubit]):
    """Convert units for `qibosoq.abstract.Sweeper` considering also LOs."""
    for idx, jdx in enumerate(sweeper.indexes):
        parameter = sweeper.parameters[idx]
        if parameter is rfsoc.Parameter.FREQUENCY:
            pulse = sequence[jdx]
            lo_frequency = pulse_lo_frequency(pulse, qubits)

            sweeper.starts[idx] = (sweeper.starts[idx] - lo_frequency) * HZ_TO_MHZ
            sweeper.stops[idx] = (sweeper.stops[idx] - lo_frequency) * HZ_TO_MHZ
        elif parameter is rfsoc.Parameter.DELAY:
            sweeper.starts[idx] = sweeper.starts[idx] * NS_TO_US
            sweeper.stops[idx] = sweeper.stops[idx] * NS_TO_US
        elif parameter is rfsoc.Parameter.RELATIVE_PHASE:
            sweeper.starts[idx] = np.degrees(sweeper.starts[idx])
            sweeper.stops[idx] = np.degrees(sweeper.stops[idx])


def convert_parameter(par: Parameter) -> rfsoc.Parameter:
    """Convert a qibolab sweeper.Parameter into a qibosoq.Parameter."""
    return getattr(rfsoc.Parameter, par.name.upper())


def convert_sweep(sweeper: Sweeper, sequence: PulseSequence, qubits: dict[int, Qubit]) -> rfsoc.Sweeper:
    """Convert `qibolab.sweeper.Sweeper` to `qibosoq.abstract.Sweeper`.

    Note that any unit conversion is not done in this function (to avoid to do it multiple times).
    Conversion will be done in `convert_units_sweeper`.
    """
    parameters = []
    starts = []
    stops = []
    indexes = []

    if sweeper.parameter is BIAS:
        for qubit in sweeper.qubits:
            parameters.append(rfsoc.Parameter.BIAS)
            indexes.append(list(qubits.values()).index(qubit))

            base_value = qubit.flux.offset
            values = sweeper.get_values(base_value)
            starts.append(values[0])
            stops.append(values[-1])

        if max(np.abs(starts)) > 1 or max(np.abs(stops)) > 1:
            raise ValueError("Sweeper amplitude is set to reach values higher than 1")
    else:
        for pulse in sweeper.pulses:
            idx_sweep = sequence.index(pulse)
            indexes.append(idx_sweep)
            base_value = getattr(pulse, sweeper.parameter.name)
            if idx_sweep != 0 and sweeper.parameter is START:
                # do the conversion from start to delay
                base_value = base_value - sequence[idx_sweep - 1].start
            values = sweeper.get_values(base_value)
            starts.append(values[0])
            stops.append(values[-1])

            if sweeper.parameter is START:
                parameters.append(rfsoc.Parameter.DELAY)
            elif sweeper.parameter is DURATION:
                parameters.append(rfsoc.Parameter.DURATION)
                delta_start = values[0] - base_value
                delta_stop = values[-1] - base_value

                if len(sequence) > idx_sweep + 1:
                    # if duration-swept pulse is not last
                    indexes.append(idx_sweep + 1)
                    t0 = sequence[idx_sweep + 1].start - sequence[idx_sweep].start
                    parameters.append(rfsoc.Parameter.DELAY)
                    starts.append(t0 + delta_start)
                    stops.append(t0 + delta_stop)
            else:
                parameters.append(convert_parameter(sweeper.parameter))

    return rfsoc.Sweeper(
        parameters=parameters,
        indexes=indexes,
        starts=starts,
        stops=stops,
        expts=len(sweeper.values),
    )


@dataclass
class RFSoCPort(Port):
    """Port object of the RFSoC."""

    name: int
    """DAC number."""
    offset: float = 0.0
    """Amplitude factor for biasing."""


class QibosoqError(RuntimeError):
    """Exception raised when qibosoq server encounters an error.

    Attributes:
    message -- The error message received from the server (qibosoq)
    """


class RFSoC(Controller):
    """Instrument object for controlling RFSoC FPGAs.

    The two way of executing pulses are with ``play`` (for arbitrary
    qibolab ``PulseSequence``) or with ``sweep`` that execute a
    ``PulseSequence`` object with one or more ``Sweeper``.

    Attributes:
        cfg (rfsoc.Config): Configuration dictionary required for pulse execution.
    """

    PortType = RFSoCPort

    def __init__(self, name: str, address: str, port: int):
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

    def connect(self):
        """Empty method to comply with Instrument interface."""

    def start(self):
        """Empty method to comply with Instrument interface."""

    def stop(self):
        """Empty method to comply with Instrument interface."""

    def disconnect(self):
        """Empty method to comply with Instrument interface."""

    def setup(self):
        """Empty deprecated method."""

    def _execute_pulse_sequence(
        self, sequence: PulseSequence, qubits: dict[int, Qubit], average: bool, opcode: rfsoc.OperationCode
    ) -> tuple[list, list]:
        """Prepare the commands dictionary to send to the qibosoq server.

        Args:
            sequence (`qibolab.pulses.PulseSequence`): arbitrary PulseSequence object to execute
            qubits: list of qubits (`qibolab.platforms.abstract.Qubit`) of the platform in the form of a dictionary
            average: if True returns averaged results, otherwise single shots
            opcode: can be `rfsoc.OperationCode.EXECUTE_PULSE_SEQUENCE` or `rfsoc.OperationCode.EXECUTE_PULSE_SEQUENCE_RAW`
        Returns:
            Lists of I and Q value measured
        """
        server_commands = {
            "operation_code": opcode,
            "cfg": asdict(self.cfg),
            "sequence": convert_pulse_sequence(sequence, qubits),
            "qubits": [asdict(convert_qubit(qubits[idx])) for idx in qubits],
            "readouts_per_experiment": len(sequence.ro_pulses),
            "average": average,
        }
        return self._open_connection(self.host, self.port, server_commands)

    def _execute_sweeps(
        self,
        sequence: PulseSequence,
        qubits: dict[int, Qubit],
        sweepers: list[rfsoc.Sweeper],
        average: bool,
    ) -> tuple[list, list]:
        """Prepare the commands dictionary to send to the qibosoq server.

        Args:
            sequence (`qibolab.pulses.PulseSequence`): arbitrary PulseSequence object to execute
            qubits: list of qubits (`qibolab.platforms.abstract.Qubit`) of the platform in the form of a dictionary
            sweepers: list of `qibosoq.abstract.Sweeper` objects
            average: if True returns averaged results, otherwise single shots
        Returns:
            Lists of I and Q value measured
        """
        for sweeper in sweepers:
            convert_units_sweeper(sweeper, sequence, qubits)
        server_commands = {
            "operation_code": rfsoc.OperationCode.EXECUTE_SWEEPS,
            "cfg": asdict(self.cfg),
            "sequence": convert_pulse_sequence(sequence, qubits),
            "qubits": [asdict(convert_qubit(qubits[idx])) for idx in qubits],
            "sweepers": [asdict(sweeper) for sweeper in sweepers],
            "readouts_per_experiment": len(sequence.ro_pulses),
            "average": average,
        }
        return self._open_connection(self.host, self.port, server_commands)

    @staticmethod
    def _open_connection(host: str, port: int, server_commands: dict):
        """Send to the server the commands needed for execution.

           The communication protocol is:
            * convert the dictionary containing all needed information in json
            * send to the server the length in byte of the encoded dictionary
            * the server now will wait for that number of bytes
            * send the  encoded dictionary
            * wait for response (arbitray number of bytes)
        Returns:
            Lists of I and Q value measured
        Raise:
            Exception: if the server encounters and error, the same error is raised here
        """
        # open a connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            msg_encoded = bytes(json.dumps(server_commands), "utf-8")
            # first send 4 bytes with the length of the message
            sock.send(len(msg_encoded).to_bytes(4, "big"))
            sock.send(msg_encoded)
            # wait till the server is sending
            received = bytearray()
            while True:
                tmp = sock.recv(4096)
                if not tmp:
                    break
                received.extend(tmp)
        results = json.loads(received.decode("utf-8"))
        if isinstance(results, str) and "Error" in results:
            raise QibosoqError(results)
        return results["i"], results["q"]

    def play(
        self,
        qubits: dict[int, Qubit],
        sequence: PulseSequence,
        execution_parameters: ExecutionParameters,
    ) -> dict[str, Union[IntegratedResults, SampleResults]]:
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
        self.validate_input_command(sequence, execution_parameters, sweep=False)
        self.update_cfg(execution_parameters)

        if execution_parameters.acquisition_type is AcquisitionType.DISCRIMINATION:
            average = False
        else:
            average = execution_parameters.averaging_mode is AveragingMode.CYCLIC

        if execution_parameters.acquisition_type is AcquisitionType.RAW:
            opcode = rfsoc.OperationCode.EXECUTE_PULSE_SEQUENCE_RAW
        else:
            opcode = rfsoc.OperationCode.EXECUTE_PULSE_SEQUENCE
        toti, totq = self._execute_pulse_sequence(sequence, qubits, average, opcode)

        results = {}
        adc_chs = np.unique([qubits[p.qubit].feedback.port.name for p in sequence.ro_pulses])

        for j, channel in enumerate(adc_chs):
            for i, ro_pulse in enumerate(sequence.ro_pulses.get_qubit_pulses(channel)):
                i_pulse = np.array(toti[j][i])
                q_pulse = np.array(totq[j][i])

                if execution_parameters.acquisition_type is AcquisitionType.DISCRIMINATION:
                    discriminated_shots = self.classify_shots(i_pulse, q_pulse, qubits[ro_pulse.qubit])
                    if execution_parameters.averaging_mode is AveragingMode.CYCLIC:
                        discriminated_shots = np.mean(discriminated_shots, axis=0)
                    result = execution_parameters.results_type(discriminated_shots)
                else:
                    result = execution_parameters.results_type(i_pulse + 1j * q_pulse)
                results[ro_pulse.qubit] = results[ro_pulse.serial] = result

        return results

    def play_sequences(self, qubits, sequence, options):
        raise NotImplementedError

    @staticmethod
    def validate_input_command(sequence: PulseSequence, execution_parameters: ExecutionParameters, sweep: bool):
        """Check if sequence and execution_parameters are supported."""
        if any(pulse.duration < 10 for pulse in sequence):
            raise ValueError("The minimum pulse length supported is 10 ns")
        if execution_parameters.acquisition_type is AcquisitionType.RAW:
            if sweep:
                raise NotImplementedError("Raw data acquisition is not compatible with sweepers")
            if len(sequence.ro_pulses) != 1:
                raise NotImplementedError("Raw data acquisition is compatible only with a single readout")
            if execution_parameters.averaging_mode is not AveragingMode.CYCLIC:
                raise NotImplementedError("Raw data acquisition can only be averaged")
        if execution_parameters.fast_reset:
            raise NotImplementedError("Fast reset is not supported")

    def update_cfg(self, execution_parameters: ExecutionParameters):
        """Update rfsoc.Config object with new parameters."""
        if execution_parameters.nshots is not None:
            self.cfg.reps = execution_parameters.nshots
        if execution_parameters.relaxation_time is not None:
            self.cfg.repetition_duration = execution_parameters.relaxation_time * NS_TO_US

    def classify_shots(self, i_values: list[float], q_values: list[float], qubit: Qubit) -> list[float]:
        """Classify IQ values using qubit threshold and rotation_angle if available in runcard."""
        if qubit.iq_angle is None or qubit.threshold is None:
            return None
        angle = qubit.iq_angle
        threshold = qubit.threshold

        rotated = np.cos(angle) * np.array(i_values) - np.sin(angle) * np.array(q_values)
        shots = np.heaviside(np.array(rotated) - threshold, 0)
        if isinstance(shots, float):
            return [shots]
        return shots

    def play_sequence_in_sweep_recursion(
        self,
        qubits: list[Qubit],
        sequence: PulseSequence,
        or_sequence: PulseSequence,
        execution_parameters: ExecutionParameters,
    ) -> dict[str, Union[IntegratedResults, SampleResults]]:
        """Last recursion layer, if no sweeps are present.

        After playing the sequence, the resulting dictionary keys need
        to be converted to the correct values.
        Even indexes correspond to qubit number and are not changed.
        Odd indexes correspond to readout pulses serials and are convert
        to match the original sequence (of the sweep) and not the one just executed.
        """
        res = self.play(qubits, sequence, execution_parameters)
        newres = {}
        serials = [pulse.serial for pulse in or_sequence.ro_pulses]
        for idx, key in enumerate(res):
            if idx % 2 == 1:
                newres[serials[idx // 2]] = res[key]
            else:
                newres[key] = res[key]

        return newres

    def recursive_python_sweep(
        self,
        qubits: list[Qubit],
        sequence: PulseSequence,
        or_sequence: PulseSequence,
        *sweepers: rfsoc.Sweeper,
        average: bool,
        execution_parameters: ExecutionParameters,
    ) -> dict[str, Union[IntegratedResults, SampleResults]]:
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
            average (bool): if True averages on nshots
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
            return self.play_sequence_in_sweep_recursion(qubits, sequence, or_sequence, execution_parameters)

        if not self.get_if_python_sweep(sequence, *sweepers):
            toti, totq = self._execute_sweeps(sequence, qubits, sweepers, average)
            res = self.convert_sweep_results(or_sequence, qubits, toti, totq, execution_parameters)
            return res

        sweeper = sweepers[0]
        values = []
        for idx, _ in enumerate(sweeper.indexes):
            val = np.linspace(sweeper.starts[idx], sweeper.stops[idx], sweeper.expts)
            values.append(val)

        results = {}
        for idx in range(sweeper.expts):
            # update values
            for jdx, kdx in enumerate(sweeper.indexes):
                sweeper_parameter = sweeper.parameters[jdx]
                if sweeper_parameter is rfsoc.Parameter.BIAS:
                    qubits[kdx].flux.bias = values[jdx][idx]
                elif sweeper_parameter in rfsoc.Parameter.variants(
                    {
                        "amplitude",
                        "frequency",
                        "relative_phase",
                        "duration",
                    }
                ):
                    setattr(sequence[kdx], sweeper_parameter.name.lower(), values[jdx][idx])
                elif sweeper is rfsoc.Parameter.DELAY:
                    sequence[kdx].start_delay = values[jdx][idx]

            res = self.recursive_python_sweep(
                qubits, sequence, or_sequence, *sweepers[1:], average=average, execution_parameters=execution_parameters
            )
            results = self.merge_sweep_results(results, res)
        return results  # already in the right format

    @staticmethod
    def merge_sweep_results(
        dict_a: dict[str, Union[IntegratedResults, SampleResults]],
        dict_b: dict[str, Union[IntegratedResults, SampleResults]],
    ) -> dict[str, Union[IntegratedResults, SampleResults]]:
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
                dict_a[serial] = dict_a[serial] + dict_b[serial]
            else:
                dict_a[serial] = dict_b[serial]
        return dict_a

    def get_if_python_sweep(self, sequence: PulseSequence, *sweepers: rfsoc.Sweeper) -> bool:
        """Check if a sweeper must be run with python loop or on hardware.

        To be run on qick internal loop a sweep must:
            * not be on the readout frequency
            * not be a duration sweeper
            * only one pulse per channel supported

        Args:
            sequence (`qibolab.pulses.PulseSequence`). Pulse sequence to play.
            *sweepers (`qibosoq.abstract.Sweeper`): Sweeper objects.
        Returns:
            A boolean value true if the sweeper must be executed by python
            loop, false otherwise
        """
        for sweeper in sweepers:
            for sweep_idx, parameter in enumerate(sweeper.parameters):
                if parameter is rfsoc.Parameter.BIAS:
                    continue
                if parameter is rfsoc.Parameter.DURATION:
                    return True

                is_freq = parameter is rfsoc.Parameter.FREQUENCY
                is_ro = sequence[sweeper.indexes[sweep_idx]].type == PulseType.READOUT

                # if it's a sweep on the readout freq do a python sweep
                if is_freq and is_ro:
                    return True
            if parameter is rfsoc.Parameter.DELAY:
                continue
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
        qubits: list[Qubit],
        toti: list[float],
        totq: list[float],
        execution_parameters: ExecutionParameters,
    ) -> dict[str, Union[IntegratedResults, SampleResults]]:
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
            adc_ro = [pulse for pulse in original_ro if qubits[pulse.qubit].feedback.port.name == k_val]
            for i, (ro_pulse, original_ro_pulse) in enumerate(zip(adc_ro, original_ro)):
                i_vals = np.array(toti[k][i])
                q_vals = np.array(totq[k][i])

                if execution_parameters.acquisition_type is AcquisitionType.DISCRIMINATION:
                    average = False
                else:
                    average = execution_parameters.averaging_mode is AveragingMode.CYCLIC

                if not average:
                    i_vals = np.reshape(i_vals, (self.cfg.reps, *i_vals.shape[:-1]))
                    q_vals = np.reshape(q_vals, (self.cfg.reps, *q_vals.shape[:-1]))

                if execution_parameters.acquisition_type is AcquisitionType.DISCRIMINATION:
                    qubit = qubits[original_ro_pulse.qubit]
                    discriminated_shots = self.classify_shots(i_vals, q_vals, qubit)
                    if execution_parameters.averaging_mode is AveragingMode.CYCLIC:
                        discriminated_shots = np.mean(discriminated_shots, axis=0)
                    result = execution_parameters.results_type(discriminated_shots)
                else:
                    result = execution_parameters.results_type(i_vals + 1j * q_vals)

                results[original_ro_pulse.qubit] = results[ro_pulse.serial] = result
        return results

    def sweep(
        self,
        qubits: dict[int, Qubit],
        sequence: PulseSequence,
        execution_parameters: ExecutionParameters,
        *sweepers: Sweeper,
    ) -> dict[str, Union[IntegratedResults, SampleResults]]:
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
        self.validate_input_command(sequence, execution_parameters, sweep=True)
        self.update_cfg(execution_parameters)

        if execution_parameters.acquisition_type is AcquisitionType.DISCRIMINATION:
            average = False
        else:
            average = execution_parameters.averaging_mode is AveragingMode.CYCLIC

        rfsoc_sweepers = [convert_sweep(sweep, sequence, qubits) for sweep in sweepers]

        sweepsequence = sequence.copy()

        bias_change = any(sweep.parameter is BIAS for sweep in sweepers)
        if bias_change:
            initial_biases = [qubits[idx].flux.offset if qubits[idx].flux is not None else None for idx in qubits]

        results = self.recursive_python_sweep(
            qubits,
            sweepsequence,
            sequence.ro_pulses,
            *rfsoc_sweepers,
            average=average,
            execution_parameters=execution_parameters,
        )

        if bias_change:
            for idx, qubit in enumerate(qubits.values()):
                if qubit.flux is not None:
                    qubit.flux.offset = initial_biases[idx]

        return results
