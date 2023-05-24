""" RFSoC FPGA driver.

This driver needs the library Qick installed
Supports the following FPGA:
 *   RFSoC 4x2
 *   ZCU111
"""

import pickle
import socket
from dataclasses import asdict
from typing import Dict, List, Tuple, Union

import numpy as np
import qibosoq.abstracts as rfsoc

from qibolab.instruments.abstract import AbstractInstrument
from qibolab.platforms.abstract import Qubit
from qibolab.pulses import Drag, Gaussian, Pulse, PulseSequence, PulseType, Rectangular
from qibolab.result import AveragedResults, ExecutionResults
from qibolab.sweeper import Parameter, Sweeper

HZ_TO_MHZ = 1e-6
NS_TO_US = 1e-3


def convert_qubit(qubit: Qubit) -> rfsoc.Qubit:
    if qubit.flux:
        dac = qubit.flux.ports[0][1]
        bias = qubit.flux.bias
    else:
        dac = None
        bias = 0.0
    return rfsoc.Qubit(bias, dac)


def convert_pulse(pulse: Pulse, qubits: Dict) -> rfsoc.Pulse:
    if pulse.shape is Rectangular:
        shape = rfsoc.Rectangular()
    elif pulse.shape is Gaussian:
        shape = rfsoc.Gaussian(pulse.shape.rel_sigma)
    elif pulse.shape is Drag:
        shape = rfsoc.Drag(pulse.shape.rel_sigma, pulse.shape.beta)

    adc = None
    if pulse.type is PulseType.DRIVE:
        type = "drive"
    elif pulse.type is PulseType.READOUT:
        type = "readout"
        adc = qubits[pulse.qubit].feedback.ports[0][1]
    elif pulse.type is PulseType.FLUX:
        type = "flux"

    dac = getattr(qubits[pulse.qubit], type).ports[0][1]

    try:
        lo_frequency = getattr(qubits[pulse.qubit], type).local_oscillator._frequency
    except NotImplementedError:
        lo_frequency = 0

    return rfsoc.Pulse(
        frequency=(pulse.frequency - lo_frequency) * HZ_TO_MHZ,
        amplitude=pulse.amplitude,
        relative_phase=np.degrees(pulse.relative_phase),
        start=pulse.start * NS_TO_US,
        duration=pulse.duration * NS_TO_US,
        shape=shape,
        dac=dac,
        adc=adc,
        name=pulse.serial,
        type=type,
    )


def convert_sweep(sweeper: Sweeper, sequence: PulseSequence, qubits: List[Qubit]) -> rfsoc.Sweeper:
    """Create a RfsocSweep oject from a Sweeper objects"""

    parameters = []
    starts = []
    stops = []
    indexes = []

    if sweeper.parameter is Parameter.bias:
        for qubit in sweeper.qubits:
            parameters.append(rfsoc.Parameter.bias)
            for idx, seq_qubit in enumerate(qubits):
                if qubit == seq_qubit:
                    indexes.append(idx)
            starts.append(sweeper.values[0] + qubits[qubit].flux.bias)
            stops.append(sweeper.values[-1] + qubits[qubit].flux.bias)

        print(stops, any(np.abs(stops)) > 1)
        if max(np.abs(starts)) > 1 or max(np.abs(stops)) > 1:
            raise ValueError("Sweeper amplitude is set to reach values higher than 1")
    else:
        for pulse in sweeper.pulses:
            for idx, seq_pulse in enumerate(sequence):
                if pulse == seq_pulse:
                    indexes.append(idx)
            if sweeper.parameter is Parameter.frequency:
                parameters.append(rfsoc.Parameter.frequency)
                starts.append(sweeper.values[0] + pulse.frequency)
                stops.append(sweeper.values[-1] + pulse.frequency)
            elif sweeper.parameter is Parameter.amplitude:
                parameters.append(rfsoc.Parameter.amplitude)
                starts.append(sweeper.values[0] * pulse.amplitude)
                stops.append(sweeper.values[-1] * pulse.amplitude)
            elif sweeper.parameter is Parameter.relative_phase:
                parameters.append(rfsoc.Parameter.relative_phase)
                starts.append(sweeper.values[0] + pulse.relative_phase)
                stops.append(sweeper.values[-1] + pulse.relative_phase)

    return rfsoc.Sweeper(
        parameter=sweeper.parameter,
        indexes=indexes,
        starts=np.array(starts),
        stops=np.array(stops),
        expts=len(sweeper.values),
    )


class RFSoC(AbstractInstrument):
    """Instrument object for controlling the RFSoC4x2 FPGA.
    Playing pulses requires first the execution of the ``setup`` function.
    The two way of executing pulses are with ``play`` (for arbitrary
    qibolab ``PulseSequence``) or with ``sweep`` that execute a
    ``PulseSequence`` object with one or more ``Sweeper``.

    Args:
        name (str): Name of the instrument instance.
    Attributes:
        cfg (rfsoc.Config): Configuration dictionary required for pulse execution.
        soc (QickSoc): ``Qick`` object needed to access system blocks.
    """

    def __init__(self, name: str, address: str):
        super().__init__(name, address=address)
        self.host, self.port = address.split(":")
        self.port = int(self.port)
        self.cfg = rfsoc.Config()  # Containes the main settings

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
        sampling_rate: int = None,
        relaxation_time: int = None,
        adc_trig_offset: int = None,
        max_gain: int = None,
    ):  # TODO rethink arguments
        """Changes the configuration of the instrument.

        Args:
            sampling_rate (int): sampling rate of the RFSoC (Hz).
            relaxation_time (int): delay before readout (ns).
            adc_trig_offset (int): single offset for all adc triggers
                                   (tproc CLK ticks).
            max_gain (int): maximum output power of the DAC (DAC units).
        """
        if sampling_rate is not None:
            self.cfg.sampling_rate = sampling_rate
        if relaxation_time is not None:
            self.cfg.repetition_duration = relaxation_time
        if adc_trig_offset is not None:
            self.cfg.adc_trig_offset = adc_trig_offset
        if max_gain is not None:
            self.cfg.max_gain = max_gain

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
            cfg: rfsoc.Config object with general settings for Qick programs
            sequence: arbitrary PulseSequence object to execute
            qubits: list of qubits of the platform
            readouts_per_experiment: number of readout pulse to execute
            average: if True returns averaged results, otherwise single shots
        Returns:
            Lists of I and Q value measured
        """
        # TODO typehint qubits is wrong, it's dictionary

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
            cfg: rfsoc.Config object with general settings for Qick programs
            sequence: arbitrary PulseSequence object to execute
            qubits: list of qubits of the platform
            sweeper: rfsoc.Sweeper object
            readouts_per_experiment: number of readout pulse to execute
            average: if True returns averaged results, otherwise single shots
        Returns:
            Lists of I and Q value measured
        """
        # TODO typehint qubits is wrong, it's dictionary

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
            * pickle the dictionary containing all needed information
            * send to the server the length in byte of the pickled dictionary
            * the server now will wait for that number of bytes
            * send the  pickled dictionary
            * wait for response (arbitray number of bytes)
        Returns:
            Lists of I and Q value measured
        Raise:
            Exception: if the server encounters and error, the same error is raised here
        """
        # open a connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            msg_encoded = pickle.dumps(server_commands)
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
        results = pickle.loads(received)
        if isinstance(results, Exception):
            raise results
        return results["i"], results["q"]

    def play(
        self,
        qubits: Dict[int, Qubit],
        sequence: PulseSequence,
        relaxation_time: int = None,
        nshots: int = None,
        average: bool = False,
        raw_adc: bool = False,
    ) -> Dict[str, ExecutionResults]:
        """Executes the sequence of instructions and retrieves readout results.
           Each readout pulse generates a separate acquisition.
           The relaxation_time and the number of shots have default values.

        Args:
            qubits (list): List of `qibolab.platforms.utils.Qubit` objects
                           passed from the platform.
            sequence (`qibolab.pulses.PulseSequence`). Pulse sequence to play.
            nshots (int): Number of repetitions (shots) of the experiment.
            relaxation_time (int): Time to wait for the qubit to relax to its
                                   ground state between shots in ns.
            raw_adc (bool): allows to acquire raw adc data
        Returns:
            A dictionary mapping the readout pulses serial and respective qubits to
            `qibolab.ExecutionResults` objects
        """

        if any(pulse.duration < 10 for pulse in sequence):
            raise ValueError("The minimum pulse length supported is 10 ns")
        if raw_adc:
            raise NotImplementedError("Raw data acquisition is not supported")

        # if new value are passed, they are updated in the config obj
        if nshots is not None:
            self.cfg.reps = nshots
        if relaxation_time is not None:
            self.cfg.repetition_duration = relaxation_time

        toti, totq = self._execute_pulse_sequence(self.cfg, sequence, qubits, len(sequence.ro_pulses), average)

        results = {}
        adc_chs = np.unique([qubits[p.qubit].feedback.ports[0][1] for p in sequence.ro_pulses])

        for j in range(len(adc_chs)):
            channel = sequence.ro_pulses[j].qubit
            for i, ro_pulse in enumerate(sequence.ro_pulses.get_qubit_pulses(channel)):
                i_pulse = np.array(toti[j][i])
                q_pulse = np.array(totq[j][i])

                serial = ro_pulse.serial

                if average:
                    results[ro_pulse.qubit] = results[serial] = AveragedResults.from_components(
                        np.array([i_pulse]), np.array([q_pulse])
                    )
                else:
                    shots = self.classify_shots(i_pulse, q_pulse, qubits[ro_pulse.qubit])
                    results[ro_pulse.qubit] = results[serial] = ExecutionResults.from_components(
                        i_pulse, q_pulse, shots
                    )

        return results

    def classify_shots(self, i_values: List[float], q_values: List[float], qubit: Qubit) -> List[float]:
        """Classify IQ values using qubit threshold and rotation_angle if available in runcard"""
        # TODO maybe move to qibosoq

        if qubit.iq_angle is None or qubit.threshold is None:
            return None
        angle = qubit.iq_angle
        threshold = qubit.threshold

        rotated = np.cos(angle) * np.array(i_values) - np.sin(angle) * np.array(q_values)
        shots = np.heaviside(np.array(rotated) - threshold, 0)
        return shots

    def recursive_python_sweep(
        self,
        qubits: List[Qubit],
        sequence: PulseSequence,
        original_ro: PulseSequence,
        *sweepers: rfsoc.Sweeper,
        average: bool,
    ) -> Dict[str, Union[AveragedResults, ExecutionResults]]:
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
            res = self.play(qubits, sequence, average=average)
            newres = {}
            serials = [pulse.serial for pulse in original_ro]
            for idx, key in enumerate(res):
                if idx % 2 == 1:
                    newres[serials[idx // 2]] = res[key]
                else:
                    newres[key] = res[key]

            return newres

        if not self.get_if_python_sweep(sequence, qubits, *sweepers):
            toti, totq = self._execute_sweeps(self.cfg, sequence, qubits, sweepers, len(sequence.ro_pulses), average)
            res = self.convert_sweep_results(original_ro, sequence, qubits, toti, totq, average)
            return res
        sweeper = sweepers[0]
        values = []
        if (
            sweeper.parameter is Parameter.frequency
            or sweeper.parameter is Parameter.amplitude
            or sweeper.parameter is Parameter.relative_phase
        ):
            for (idx,) in range(len(sweeper.indexes)):
                val = np.linspace(sweeper.starts[idx], sweeper.stops[idx], sweeper.expts)
                values.append(val)
        else:
            for idx, _ in enumerate(sweeper.indexes):
                val = np.linspace(sweeper.starts[idx], sweeper.stops[idx], sweeper.expts)
                values.append(val)

        results = {}
        for idx in range(sweeper.expts):
            # update values
            if (
                sweeper.parameter is Parameter.frequency
                or sweeper.parameter is Parameter.amplitude
                or sweeper.parameter is Parameter.relative_phase
            ):
                for jdx in range(len(sweeper.indexes)):
                    if sweeper.parameter is Parameter.frequency:
                        sequence[sweeper.indexes[jdx]].frequency = values[jdx][idx]
                    elif sweeper.parameter is Parameter.amplitude:
                        sequence[sweeper.indexes[jdx]].amplitude = values[jdx][idx]
                    elif sweeper.parameter is Parameter.relative_phase:
                        sequence[sweeper.indexes[jdx]].relative_phase = values[jdx][idx]
            else:
                for kdx, jdx in enumerate(sweeper.indexes):
                    qubits[jdx].flux.bias = values[kdx][idx]

            res = self.recursive_python_sweep(qubits, sequence, original_ro, *sweepers[1:], average=average)
            results = self.merge_sweep_results(results, res)
        return results  # already in the right format

    @staticmethod
    def merge_sweep_results(
        dict_a: Dict[str, Union[AveragedResults, ExecutionResults]],
        dict_b: Dict[str, Union[AveragedResults, ExecutionResults]],
    ) -> Dict[str, Union[AveragedResults, ExecutionResults]]:
        """Merge two dictionary mapping pulse serial to Results object.
        If dict_b has a key (serial) that dict_a does not have, simply add it,
        otherwise sum the two results (`qibolab.result.ExecutionResults`
        or `qibolab.result.AveragedResults`)

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
            * be just one sweeper
            * only one pulse per channel supported (for now)

        Args:
            sequence (`qibolab.pulses.PulseSequence`). Pulse sequence to play.
            qubits (list): List of `qibolab.platforms.utils.Qubit` objects
                           passed from the platform.
            *sweepers (`qibolab.Sweeper`): Sweeper objects.
        Returns:
            A boolean value true if the sweeper must be executed by python
            loop, false otherwise
        """

        for sweeper in sweepers:
            is_amp = sweeper.parameter is Parameter.amplitude
            is_freq = sweeper.parameter is Parameter.frequency

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
        average: bool,
    ) -> Dict[str, Union[ExecutionResults, AveragedResults]]:
        """Convert sweep res to qibolab dict res

        Args:
            original_ro (list): list of ro serials of the original sequence
            sequence (`qibolab.pulses.PulseSequence`). Pulse sequence to play.
            qubits (list): List of `qibolab.platforms.utils.Qubit` objects
                 passed from the platform.
            toti (list): i values
            totq (list): q values
            average (bool): true if the result is from averaged acquisition
        Returns:
            A dict mapping the readout pulses serial to qibolab results objects
        """
        sweep_results = {}

        adcs = np.unique([qubits[p.qubit].feedback.ports[0][1] for p in sequence.ro_pulses])
        for k, k_val in enumerate(adcs):
            results = {}
            serials = [pulse.serial for pulse in original_ro if qubits[pulse.qubit].feedback.ports[0][1] == k_val]
            for i, serial in enumerate(serials):
                i_pulse = np.array(toti[k][i])
                q_pulse = np.array(totq[k][i])

                # TODO new results
                i_pulse = i_pulse.flatten()
                q_pulse = q_pulse.flatten()
                i_pulse = i_pulse[i_pulse != 0]
                q_pulse = q_pulse[q_pulse != 0]

                if average:
                    results[sequence.ro_pulses[i].qubit] = results[serial] = AveragedResults.from_components(
                        i_pulse, q_pulse
                    )
                else:
                    qubit = qubits[sequence.ro_pulses[i].qubit]
                    shots = self.classify_shots(i_pulse, q_pulse, qubit)
                    results[sequence.ro_pulses[i].qubit] = results[serial] = ExecutionResults.from_components(
                        i_pulse, q_pulse, shots
                    )
            sweep_results = self.merge_sweep_results(sweep_results, results)

        return sweep_results

    def sweep(
        self,
        qubits: List[Qubit],
        sequence: PulseSequence,
        *sweepers: Sweeper,
        relaxation_time: int,
        nshots: int = 1000,
        average: bool = True,
    ) -> Dict[str, Union[AveragedResults, ExecutionResults]]:
        """Executes the sweep and retrieves the readout results.
        Each readout pulse generates a separate acquisition.
        The relaxation_time and the number of shots have default values.

        Args:
            qubits (list): List of `qibolab.platforms.utils.Qubit` objects
                           passed from the platform.
            sequence (`qibolab.pulses.PulseSequence`). Pulse sequence to play.
            *sweepers (`qibolab.Sweeper`): Sweeper objects.
            relaxation_time (int): Time to wait for the qubit to relax to its
                                   ground state between shots in ns.
            nshots (int): Number of repetitions (shots) of the experiment.
            average (bool): if False returns single shot measurements
        Returns:
            A dictionary mapping the readout pulses serial and respective qubits to
            results objects
        """

        # if new value are passed, they are updated in the config obj
        if nshots is not None:
            self.cfg.reps = nshots
        if relaxation_time is not None:
            self.cfg.repetition_duration = relaxation_time

        rfsoc_sweepers = [convert_sweep(sweep, sequence, qubits) for sweep in sweepers]

        sweepsequence = sequence.copy()

        original_ro = sequence.ro_pulses

        bias_change = any([sweep.parameter is Parameter.bias for sweep in sweepers])
        if bias_change:
            initial_biases = [qubits[idx].flux.bias if qubits[idx].flux is not None else None for idx in qubits]

        results = self.recursive_python_sweep(qubits, sweepsequence, original_ro, *rfsoc_sweepers, average=average)

        if bias_change:
            for idx in qubits:
                if qubits[idx].flux is not None:
                    qubits[idx].flux.bias = initial_biases[idx]

        return results


class TII_RFSOC4x2(RFSoC):
    """RFSoC object for Xilinx RFSoC4x2"""

    def __init__(self, name: str, address: str):
        """Define IP, port and rfsoc.Config"""
        super().__init__(name, address=address)
        self.host, self.port = address.split(":")
        self.port = int(self.port)
        self.cfg = rfsoc.Config()


class TII_ZCU111(RFSoC):
    """RFSoC object for Xilinx ZCU111"""

    def __init__(self, name: str, address: str):
        """Define IP, port and rfsoc.Config"""
        super().__init__(name, address=address)
        self.host, self.port = address.split(":")
        self.port = int(self.port)
        self.cfg = rfsoc.Config()
