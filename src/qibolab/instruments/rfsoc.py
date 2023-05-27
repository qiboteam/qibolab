""" RFSoC FPGA driver.

This driver needs a Qibosoq server installed and running
Tested on the following FPGA:
 *   RFSoC 4x2
 *   ZCU111
"""

import json
import socket
from dataclasses import asdict
from typing import Dict, List, Tuple, Union

import numpy as np
import qibosoq.abstracts as rfsoc

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.instruments.abstract import AbstractInstrument
from qibolab.platforms.abstract import Qubit
from qibolab.pulses import Drag, Gaussian, Pulse, PulseSequence, PulseType, Rectangular
from qibolab.result import IntegratedResults, SampleResults
from qibolab.sweeper import Parameter, Sweeper

HZ_TO_MHZ = 1e-6
NS_TO_US = 1e-3


def convert_qubit(qubit: Qubit) -> rfsoc.Qubit:
    """Convert `qibolab.platforms.abstract.Qubit` to `qibosoq.abstract.Qubit`"""
    if qubit.flux:
        return rfsoc.Qubit(qubit.flux.bias, qubit.flux.ports[0][1])
    return rfsoc.Qubit(0.0, None)


def convert_pulse(pulse: Pulse, qubits: Dict[int, Qubit]) -> rfsoc.Pulse:
    """Convert `qibolab.pulses.pulse` to `qibosoq.abstract.Pulse`"""
    pulse_type = pulse.type.name.lower()
    dac = getattr(qubits[pulse.qubit], pulse_type).ports[0][1]
    adc = qubits[pulse.qubit].feedback.ports[0][1] if pulse_type == "readout" else None

    try:
        lo_frequency = getattr(qubits[pulse.qubit], pulse_type).local_oscillator._frequency
    except NotImplementedError:
        lo_frequency = 0

    rfsoc_pulse = rfsoc.Pulse(
        frequency=(pulse.frequency - lo_frequency) * HZ_TO_MHZ,
        amplitude=pulse.amplitude,
        relative_phase=np.degrees(pulse.relative_phase),
        start=pulse.start * NS_TO_US,
        duration=pulse.duration * NS_TO_US,
        dac=dac,
        adc=adc,
        shape=None,
        name=pulse.serial,
        type=pulse_type,
    )
    if isinstance(pulse.shape, Rectangular):
        rfsoc_pulse.shape = "rectangular"
    elif isinstance(pulse.shape, Gaussian):
        rfsoc_pulse.shape = "gaussian"
        rfsoc_pulse.rel_sigma = pulse.shape.rel_sigma
    elif isinstance(pulse.shape, Drag):
        rfsoc_pulse.shape = "drag"
        rfsoc_pulse.rel_sigma = pulse.shape.rel_sigma
        rfsoc_pulse.beta = pulse.shape.beta
    return rfsoc_pulse


def convert_frequency_sweeper(sweeper: rfsoc.Sweeper, sequence: PulseSequence, qubits: Dict[int, Qubit]):
    """Converts frequencies for `qibosoq.abstract.Sweeper` considering LO and HZ_TO_MHZ"""
    for idx, jdx in enumerate(sweeper.indexes):
        if sweeper.parameter[idx] is rfsoc.Parameter.frequency:
            pulse = sequence[jdx]
            try:
                lo_frequency = getattr(qubits[pulse.qubit], pulse.type.name.lower()).local_oscillator._frequency
            except NotImplementedError:
                lo_frequency = 0

            sweeper.starts[idx] = (sweeper.starts[idx] - lo_frequency) * HZ_TO_MHZ
            sweeper.stops[idx] = (sweeper.stops[idx] - lo_frequency) * HZ_TO_MHZ


def convert_sweep(sweeper: Sweeper, sequence: PulseSequence, qubits: Dict[int, Qubit]) -> rfsoc.Sweeper:
    """Convert `qibolab.sweeper.Sweeper` to `qibosoq.abstract.Sweeper`"""

    parameters = []
    starts = []
    stops = []
    indexes = []

    if sweeper.parameter is Parameter.bias:
        for qubit in sweeper.qubits:
            parameters.append(rfsoc.Parameter.bias)
            indexes.append(list(qubits.values().index(qubit)))

            starts.append(sweeper.values[0] + qubit.flux.bias)
            stops.append(sweeper.values[-1] + qubit.flux.bias)

        if max(np.abs(starts)) > 1 or max(np.abs(stops)) > 1:
            raise ValueError("Sweeper amplitude is set to reach values higher than 1")
    else:
        for pulse in sweeper.pulses:
            indexes.append(sequence.index(pulse))

            name = sweeper.parameter.name
            parameters.append(getattr(rfsoc.Parameter, name))
            value = getattr(pulse, name)
            if sweeper.parameter in {Parameter.frequency, Parameter.relative_phase}:
                starts.append(sweeper.values[0] + value)
            elif sweeper.parameter is Parameter.amplitude:
                starts.append(sweeper.values[0] * value)
            else:
                raise NotImplementedError(f"Sweep parameter {sweeper.parameter} not implemented")

    return rfsoc.Sweeper(
        parameter=parameters,
        indexes=indexes,
        starts=starts,
        stops=stops,
        expts=len(sweeper.values),
    )


class RFSoC(AbstractInstrument):
    """Instrument object for controlling RFSoC FPGAs.
    The two way of executing pulses are with ``play`` (for arbitrary
    qibolab ``PulseSequence``) or with ``sweep`` that execute a
    ``PulseSequence`` object with one or more ``Sweeper``.

    Attributes:
        cfg (rfsoc.Config): Configuration dictionary required for pulse execution.
    """

    def __init__(self, name: str, address: str, port: int):
        """__init__
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
        """Empty method to comply with AbstractInstrument interface."""

    def start(self):
        """Empty method to comply with AbstractInstrument interface."""

    def stop(self):
        """Empty method to comply with AbstractInstrument interface."""

    def disconnect(self):
        """Empty method to comply with AbstractInstrument interface."""

    def setup(
        self,
        relaxation_time: int = None,
        adc_trig_offset: int = None,
    ):
        """Changes the configuration of the instrument.

        Args:
            relaxation_time (int): delay before readout (ns).
            adc_trig_offset (int): single offset for all adc triggers
                                   (tproc CLK ticks).
        """
        if relaxation_time is not None:
            self.cfg.repetition_duration = relaxation_time * NS_TO_US
        if adc_trig_offset is not None:
            self.cfg.adc_trig_offset = adc_trig_offset

    def _execute_pulse_sequence(
        self,
        cfg: rfsoc.Config,
        sequence: PulseSequence,
        qubits: Dict[int, Qubit],
        readouts_per_experiment: int,
        average: bool,
    ) -> Tuple[list, list]:
        """Prepares the dictionary to send to the qibosoq server in order
           to execute a PulseSequence.

        Args:
            cfg (`qibosoq.abstract.Config`): general settings for Qick programs
            sequence (`qibolab.pulses.PulseSequence`): arbitrary PulseSequence object to execute
            qubits: list of qubits (`qibolab.platforms.abstract.Qubit`) of the platform in the form of a dictionary
            readouts_per_experiment: number of readout pulse to execute
            average: if True returns averaged results, otherwise single shots
        Returns:
            Lists of I and Q value measured
        """

        server_commands = {
            "operation_code": "execute_pulse_sequence",
            "cfg": asdict(cfg),
            "sequence": [asdict(convert_pulse(pulse, qubits)) for pulse in sequence],
            "qubits": [asdict(convert_qubit(qubits[idx])) for idx in qubits],
            "readouts_per_experiment": readouts_per_experiment,
            "average": average,
        }
        return self._open_connection(self.host, self.port, server_commands)

    def _execute_sweeps(
        self,
        cfg: rfsoc.Config,
        sequence: PulseSequence,
        qubits: Dict[int, Qubit],
        sweepers: List[rfsoc.Sweeper],
        readouts_per_experiment: int,
        average: bool,
    ) -> Tuple[list, list]:
        """Prepares the dictionary to send to the qibosoq server in order
           to execute a sweep.

        Args:
            cfg (`qibosoq.abstract.Config`): general settings for Qick programs
            sequence (`qibolab.pulses.PulseSequence`): arbitrary PulseSequence object to execute
            qubits: list of qubits (`qibolab.platforms.abstract.Qubit`) of the platform in the form of a dictionary
            sweepers: list of `qibosoq.abstract.Sweeper` objects
            readouts_per_experiment: number of readout pulse to execute
            average: if True returns averaged results, otherwise single shots
        Returns:
            Lists of I and Q value measured
        """

        for sweeper in sweepers:
            convert_frequency_sweeper(sweeper, sequence, qubits)
        server_commands = {
            "operation_code": "execute_sweeps",
            "cfg": asdict(cfg),
            "sequence": [asdict(convert_pulse(pulse, qubits)) for pulse in sequence],
            "qubits": [asdict(convert_qubit(qubits[idx])) for idx in qubits],
            "sweepers": [asdict(sweeper) for sweeper in sweepers],
            "readouts_per_experiment": readouts_per_experiment,
            "average": average,
        }
        return self._open_connection(self.host, self.port, server_commands)

    @staticmethod
    def _open_connection(host: str, port: int, server_commands: dict):
        """Sends to the server on board all the objects and information needed for
           executing a sweep or a pulse sequence.

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
        results = json.loads(received)
        if isinstance(results, str):
            raise RuntimeError(results)
        return results["i"], results["q"]

    def play(
        self,
        qubits: Dict[int, Qubit],
        sequence: PulseSequence,
        execution_parameters: ExecutionParameters,
    ) -> Dict[str, Union[IntegratedResults, SampleResults]]:
        """Executes the sequence of instructions and retrieves readout results.
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

        if any(pulse.duration < 10 for pulse in sequence):
            raise ValueError("The minimum pulse length supported is 10 ns")
        if execution_parameters.acquisition_type is AcquisitionType.RAW:
            raise NotImplementedError("Raw data acquisition is not supported")
        if execution_parameters.fast_reset:
            raise NotImplementedError("Fast reset is not supported")

        # if new value are passed, they are updated in the config obj
        if execution_parameters.nshots is not None:
            self.cfg.reps = execution_parameters.nshots
        if execution_parameters.relaxation_time is not None:
            self.cfg.repetition_duration = execution_parameters.relaxation_time * NS_TO_US

        if execution_parameters.acquisition_type is AcquisitionType.DISCRIMINATION:
            average = False
        else:
            average = execution_parameters.averaging_mode is AveragingMode.CYCLIC

        toti, totq = self._execute_pulse_sequence(self.cfg, sequence, qubits, len(sequence.ro_pulses), average)

        results = {}
        adc_chs = np.unique([qubits[p.qubit].feedback.ports[0][1] for p in sequence.ro_pulses])

        for j in range(len(adc_chs)):
            channel = sequence.ro_pulses[j].qubit
            for i, ro_pulse in enumerate(sequence.ro_pulses.get_qubit_pulses(channel)):
                i_pulse = np.array(toti[j][i])
                q_pulse = np.array(totq[j][i])

                serial = ro_pulse.serial

                if execution_parameters.acquisition_type is AcquisitionType.DISCRIMINATION:
                    discriminated_shots = self.classify_shots(i_pulse, q_pulse, qubits[ro_pulse.qubit])
                    if execution_parameters.averaging_mode is AveragingMode.CYCLIC:
                        discriminated_shots = np.mean(discriminated_shots, axis=0)
                    result = execution_parameters.results_type(discriminated_shots)
                else:
                    result = execution_parameters.results_type(i_pulse + 1j * q_pulse)
                results[ro_pulse.qubit] = results[serial] = result

        return results

    def classify_shots(self, i_values: List[float], q_values: List[float], qubit: Qubit) -> List[float]:
        """Classify IQ values using qubit threshold and rotation_angle if available in runcard"""

        if qubit.iq_angle is None or qubit.threshold is None:
            return None
        angle = qubit.iq_angle
        threshold = qubit.threshold

        rotated = np.cos(angle) * np.array(i_values) - np.sin(angle) * np.array(q_values)
        shots = np.heaviside(np.array(rotated) - threshold, 0)
        if isinstance(shots, float):
            return [shots]
        return shots

    def recursive_python_sweep(
        self,
        qubits: List[Qubit],
        sequence: PulseSequence,
        or_sequence: PulseSequence,
        *sweepers: rfsoc.Sweeper,
        average: bool,
        execution_parameters: ExecutionParameters,
    ) -> Dict[str, Union[IntegratedResults, SampleResults]]:
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
        Raises:
            NotImplementedError: if a sweep refers to more than one pulse.
            NotImplementedError: if a sweep refers to a parameter different
                                 from frequency or amplitude.
        """
        # If there are no sweepers run ExecutePulseSequence acquisition.
        # Last layer for recursion.

        if len(sweepers) == 0:
            res = self.play(qubits, sequence, execution_parameters)
            newres = {}
            serials = [pulse.serial for pulse in or_sequence.ro_pulses]
            for idx, key in enumerate(res):
                if idx % 2 == 1:
                    newres[serials[idx // 2]] = res[key]
                else:
                    newres[key] = res[key]

            return newres

        if not self.get_if_python_sweep(sequence, qubits, *sweepers):
            toti, totq = self._execute_sweeps(self.cfg, sequence, qubits, sweepers, len(sequence.ro_pulses), average)
            res = self.convert_sweep_results(or_sequence, sequence, qubits, toti, totq, execution_parameters)
            return res

        sweeper = sweepers[0]
        values = []
        for idx, _ in enumerate(sweeper.indexes):
            val = np.linspace(sweeper.starts[idx], sweeper.stops[idx], sweeper.expts)
            values.append(val)

        results = {}
        for idx in range(sweeper.expts):
            # update values
            if sweeper.parameter[0] in {
                rfsoc.Parameter.amplitude,
                rfsoc.Parameter.frequency,
                rfsoc.Parameter.relative_phase,
            }:
                for jdx, _ in enumerate(sweeper.indexes):
                    if sweeper.parameter[jdx] is rfsoc.Parameter.frequency:
                        sequence[sweeper.indexes[jdx]].frequency = values[jdx][idx]
                    elif sweeper.parameter[jdx] is rfsoc.Parameter.amplitude:
                        sequence[sweeper.indexes[jdx]].amplitude = values[jdx][idx]
                    elif sweeper.parameter[jdx] is rfsoc.Parameter.relative_phase:
                        sequence[sweeper.indexes[jdx]].relative_phase = values[jdx][idx]
            else:
                for kdx, jdx in enumerate(sweeper.indexes):
                    qubits[jdx].flux.bias = values[kdx][idx]

            res = self.recursive_python_sweep(
                qubits, sequence, or_sequence, *sweepers[1:], average=average, execution_parameters=execution_parameters
            )
            results = self.merge_sweep_results(results, res)
        return results  # already in the right format

    @staticmethod
    def merge_sweep_results(
        dict_a: Dict[str, Union[IntegratedResults, SampleResults]],
        dict_b: Dict[str, Union[IntegratedResults, SampleResults]],
    ) -> Dict[str, Union[IntegratedResults, SampleResults]]:
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

    def get_if_python_sweep(self, sequence: PulseSequence, qubits: List[Qubit], *sweepers: rfsoc.Sweeper) -> bool:
        """Check if a sweeper must be run with python loop or on hardware.

        To be run on qick internal loop a sweep must:
            * not be on the readout frequency
            * only one pulse per channel supported

        Args:
            sequence (`qibolab.pulses.PulseSequence`). Pulse sequence to play.
            qubits (list): List of `qibolab.platforms.utils.Qubit` objects
                           passed from the platform.
            *sweepers (`qibosoq.abstract.Sweeper`): Sweeper objects.
        Returns:
            A boolean value true if the sweeper must be executed by python
            loop, false otherwise
        """

        for sweeper in sweepers:
            is_amp = sweeper.parameter[0] is rfsoc.Parameter.amplitude
            is_freq = sweeper.parameter[0] is rfsoc.Parameter.frequency

            if is_freq or is_amp:
                is_ro = sequence[sweeper.indexes[0]].type == PulseType.READOUT
                # if it's a sweep on the readout freq do a python sweep
                if is_freq and is_ro:
                    return True

                # check if the sweeped pulse is the first and only on the DAC channel
                for idx in sweeper.indexes:
                    sweep_pulse = sequence[idx]
                    already_pulsed = []
                    for pulse in sequence:
                        pulse_q = qubits[pulse.qubit]
                        pulse_is_ro = pulse.type == PulseType.READOUT
                        pulse_ch = pulse_q.readout.ports[0][1] if pulse_is_ro else pulse_q.drive.ports[0][1]

                        if pulse_ch in already_pulsed and pulse == sweep_pulse:
                            return True
                        already_pulsed.append(pulse_ch)
        # if all passed, do a firmware sweep
        return False

    def convert_sweep_results(
        self,
        original_ro: PulseSequence,
        sequence: PulseSequence,
        qubits: List[Qubit],
        toti: List[float],
        totq: List[float],
        execution_parameters: ExecutionParameters,
    ) -> Dict[str, Union[IntegratedResults, SampleResults]]:
        """Convert sweep res to qibolab dict res

        Args:
            original_ro (`qibolab.pulses.PulseSequence`): Original PulseSequence
            sequence (`qibolab.pulses.PulseSequence`): Pulse sequence to play.
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

        adcs = np.unique([qubits[p.qubit].feedback.ports[0][1] for p in sequence.ro_pulses])
        for k, k_val in enumerate(adcs):
            adc_ro = [pulse for pulse in sequence.ro_pulses if qubits[pulse.qubit].feedback.ports[0][1] == k_val]
            for i, ro_pulse in enumerate(adc_ro):
                i_vals = np.array(toti[k][i])
                q_vals = np.array(totq[k][i])

                if execution_parameters.acquisition_type is AcquisitionType.DISCRIMINATION:
                    average = False
                else:
                    average = execution_parameters.averaging_mode is AveragingMode.CYCLIC

                if not average:
                    shape = i_vals.shape
                    i_vals = np.reshape(i_vals, (self.cfg.reps, *shape[:-1]))
                    q_vals = np.reshape(q_vals, (self.cfg.reps, *shape[:-1]))

                if execution_parameters.acquisition_type is AcquisitionType.DISCRIMINATION:
                    qubit = qubits[sequence.ro_pulses[i].qubit]
                    discriminated_shots = self.classify_shots(i_vals, q_vals, qubit)
                    if execution_parameters.averaging_mode is AveragingMode.CYCLIC:
                        discriminated_shots = np.mean(discriminated_shots, axis=0)
                    result = execution_parameters.results_type(discriminated_shots)
                else:
                    result = execution_parameters.results_type(i_vals + 1j * q_vals)

                results[sequence.ro_pulses[i].qubit] = results[ro_pulse.serial] = result
        return results

    def sweep(
        self,
        qubits: Dict[int, Qubit],
        sequence: PulseSequence,
        execution_parameters: ExecutionParameters,
        *sweepers: Sweeper,
    ) -> Dict[str, Union[IntegratedResults, SampleResults]]:
        """Executes the sweep and retrieves the readout results.
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

        if execution_parameters.acquisition_type is AcquisitionType.RAW:
            raise NotImplementedError("Raw data acquisition is not supported")
        if execution_parameters.fast_reset:
            raise NotImplementedError("Fast reset is not supported")

        # if new value are passed, they are updated in the config obj
        if execution_parameters.nshots is not None:
            self.cfg.reps = execution_parameters.nshots
        if execution_parameters.relaxation_time is not None:
            self.cfg.repetition_duration = execution_parameters.relaxation_time * NS_TO_US

        if execution_parameters.acquisition_type is AcquisitionType.DISCRIMINATION:
            average = False
        else:
            average = execution_parameters.averaging_mode is AveragingMode.CYCLIC

        rfsoc_sweepers = [convert_sweep(sweep, sequence, qubits) for sweep in sweepers]

        sweepsequence = sequence.copy()

        bias_change = any([sweep.parameter is Parameter.bias for sweep in sweepers])
        if bias_change:
            initial_biases = [qubits[idx].flux.bias if qubits[idx].flux is not None else None for idx in qubits]

        results = self.recursive_python_sweep(
            qubits,
            sweepsequence,
            sequence.ro_pulses,
            *rfsoc_sweepers,
            average=average,
            execution_parameters=execution_parameters,
        )

        if bias_change:
            for idx in qubits:
                if qubits[idx].flux is not None:
                    qubits[idx].flux.bias = initial_biases[idx]

        return results
