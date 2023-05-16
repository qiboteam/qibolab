""" RFSoC FPGA driver.

This driver needs the library Qick installed
Supports the following FPGA:
 *   RFSoC 4x2
"""

import pickle
import socket
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from qibolab.instruments.abstract import AbstractInstrument
from qibolab.platforms.abstract import Qubit
from qibolab.pulses import PulseSequence, PulseType
from qibolab.result import AveragedResults, ExecutionResults
from qibolab.sweeper import Parameter, Sweeper


@dataclass
class QickProgramConfig:
    sampling_rate: int = 9_830_400_000
    repetition_duration: int = 100_000
    adc_trig_offset: int = 200
    max_gain: int = 32_000
    reps: int = 1000
    expts: Optional[int] = None


class TII_RFSOC4x2(AbstractInstrument):
    """Instrument object for controlling the RFSoC4x2 FPGA.
    Playing pulses requires first the execution of the ``setup`` function.
    The two way of executing pulses are with ``play`` (for arbitrary
    qibolab ``PulseSequence``) or with ``sweep`` that execute a
    ``PulseSequence`` object with one or more ``Sweeper``.

    Args:
        name (str): Name of the instrument instance.
    Attributes:
        cfg (QickProgramConfig): Configuration dictionary required for pulse execution.
        soc (QickSoc): ``Qick`` object needed to access system blocks.
    """

    def __init__(self, name: str, address: str):
        super().__init__(name, address=address)
        self.host, self.port = address.split(":")
        self.port = int(self.port)
        self.cfg = QickProgramConfig()  # Containes the main settings

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
    ):
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
        cfg: QickProgramConfig,
        sequence: PulseSequence,
        qubits: List[Qubit],
        readouts_per_experiment: int,
        average: bool,
    ) -> Tuple[list, list]:
        """Prepares the dictionary to send to the qibosoq server in order
           to execute a PulseSequence.

        Args:
            cfg: QickProgramConfig object with general settings for Qick programs
            sequence: arbitrary PulseSequence object to execute
            qubits: list of qubits of the platform
            readouts_per_experiment: number of readout pulse to execute
            average: if True returns averaged results, otherwise single shots
        Returns:
            Lists of I and Q value measured
        """

        server_commands = {
            "operation_code": "execute_pulse_sequence",
            "cfg": cfg,
            "sequence": sequence,
            "qubits": qubits,
            "readouts_per_experiment": readouts_per_experiment,
            "average": average,
        }
        return self._open_connection(self.host, self.port, server_commands)

    def _execute_single_sweep(
        self,
        cfg: QickProgramConfig,
        sequence: PulseSequence,
        qubits: List[Qubit],
        sweeper: Sweeper,
        readouts_per_experiment: int,
        average: bool,
    ) -> Tuple[list, list]:
        """Prepares the dictionary to send to the qibosoq server in order
           to execute a sweep.

        Args:
            cfg: QickProgramConfig object with general settings for Qick programs
            sequence: arbitrary PulseSequence object to execute
            qubits: list of qubits of the platform
            sweeper: Sweeper object
            readouts_per_experiment: number of readout pulse to execute
            average: if True returns averaged results, otherwise single shots
        Returns:
            Lists of I and Q value measured
        """

        server_commands = {
            "operation_code": "execute_single_sweep",
            "cfg": cfg,
            "sequence": sequence,
            "qubits": qubits,
            "sweeper": sweeper,
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
        qubits: List[Qubit],
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

        if raw_adc:
            raise NotImplementedError("Raw data acquisition is not supported")

        # if new value are passed, they are updated in the config obj
        if nshots is not None:
            self.cfg.reps = nshots
        if relaxation_time is not None:
            self.cfg.repetition_duration = relaxation_time

        toti, totq = self._execute_pulse_sequence(self.cfg, sequence, qubits, len(sequence.ro_pulses), average)

        results = {}
        adcs = np.unique([qubits[p.qubit].feedback.ports[0][1] for p in sequence.ro_pulses])
        for j in range(len(adcs)):
            for i, ro_pulse in enumerate(sequence.ro_pulses):
                i_pulse = np.array(toti[j][i])
                q_pulse = np.array(totq[j][i])

                serial = ro_pulse.serial

                if average:
                    results[ro_pulse.qubit] = results[serial] = AveragedResults.from_components(i_pulse, q_pulse)
                else:
                    shots = self.classify_shots(i_pulse, q_pulse, qubits[ro_pulse.qubit])
                    results[ro_pulse.qubit] = results[serial] = ExecutionResults.from_components(
                        i_pulse, q_pulse, shots
                    )

        return results

    def classify_shots(self, i_values: List[float], q_values: List[float], qubit: Qubit) -> List[float]:
        """Classify IQ values using qubit threshold and rotation_angle if available in runcard"""

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
        or_sequence: PulseSequence,
        *sweepers: Sweeper,
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
        # gets a list containing the original sequence output serials
        original_ro = [ro.serial for ro in or_sequence.ro_pulses]

        # If there are no sweepers run ExecutePulseSequence acquisition.
        # Last layer for recursion.
        if len(sweepers) == 0:
            res = self.play(qubits, sequence, average=average)
            newres = {}
            for idx, key in enumerate(res):
                newres[original_ro[idx // 2]] = res[key]
            return newres

        # If sweepers are still in queue
        else:
            # check that the first (outest) sweeper is supported
            sweeper = sweepers[0]
            if len(sweeper.pulses) > 1:
                raise NotImplementedError("Only one pulse per sweep supported")
            is_amp = sweeper.parameter == Parameter.amplitude
            is_freq = sweeper.parameter == Parameter.frequency
            if not (is_amp or is_freq):
                raise NotImplementedError("Parameter type not implemented")

            # if there is one sweeper supported by qick than use hardware sweep
            if len(sweepers) == 1 and not self.get_if_python_sweep(sequence, qubits, *sweepers):
                toti, totq = self._execute_single_sweep(
                    self.cfg, sequence, qubits, sweepers[0], len(sequence.ro_pulses), average
                )
                # convert results
                res = self.convert_sweep_results(sweepers[0], original_ro, sequence, qubits, toti, totq, average)
                return res

            # if it's not possible to execute qick sweep re-call function
            else:
                sweep_results = {}
                idx_pulse = or_sequence.index(sweeper.pulses[0])
                for val in sweeper.values:
                    if is_freq:
                        sequence[idx_pulse].frequency = val
                    elif is_amp:
                        sequence[idx_pulse].amplitude = val
                    res = self.recursive_python_sweep(qubits, sequence, or_sequence, *sweepers[1:], average=average)
                    # merge the dictionary obtained with the one already saved
                    sweep_results = self.merge_sweep_results(sweep_results, res)

        return sweep_results

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

    def get_if_python_sweep(self, sequence: PulseSequence, qubits: List[Qubit], *sweepers: Sweeper) -> bool:
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

        # if there isn't only a sweeper do a python sweep
        if len(sweepers) != 1:
            return True

        is_freq = sweepers[0].parameter is Parameter.frequency
        is_ro = sweepers[0].pulses[0].type is PulseType.READOUT
        # if it's a sweep on the readout freq do a python sweep
        if is_freq and is_ro:
            return True

        # check if the sweeped pulse is the first on the DAC channel
        already_pulsed = []
        for pulse in sequence:
            pulse_q = qubits[pulse.qubit]
            pulse_is_ro = pulse.type == PulseType.READOUT
            pulse_ch = pulse_q.readout.ports[0][1] if pulse_is_ro else pulse_q.drive.ports[0][1]

            if pulse_ch in already_pulsed:
                return True
            else:
                already_pulsed.append(pulse_ch)

        # if all passed, do a firmware sweep
        return False

    def convert_sweep_results(
        self,
        sweeper: Sweeper,
        original_ro: List[str],
        sequence: PulseSequence,
        qubits: List[Qubit],
        toti: List[float],
        totq: List[float],
        average: bool,
    ) -> Dict[str, Union[ExecutionResults, AveragedResults]]:
        """Convert sweep res to qibolab dict res

        Args:
            *sweepers (`qibolab.Sweeper`): Sweeper objects.
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
        for k in range(len(adcs)):
            for j in range(len(sweeper.values)):
                results = {}
                # add a result for every readouts pulse
                for i, serial in enumerate(original_ro):
                    i_pulse = np.array(toti[k][i][j])
                    q_pulse = np.array(totq[k][i][j])

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
                # merge new result with already saved ones
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

        # sweepers.values are modified to reflect actual sweeped values
        for sweeper in sweepers:
            if sweeper.parameter == Parameter.frequency:
                sweeper.values += sweeper.pulses[0].frequency
            elif sweeper.parameter == Parameter.amplitude:
                sweeper.values *= sweeper.pulses[0].amplitude

        sweepsequence = sequence.copy()

        results = self.recursive_python_sweep(qubits, sweepsequence, sequence, *sweepers, average=average)

        # sweepers.values are converted back to original relative values
        for sweeper in sweepers:
            if sweeper.parameter == Parameter.frequency:
                sweeper.values -= sweeper.pulses[0].frequency
            elif sweeper.parameter == Parameter.amplitude:
                sweeper.values /= sweeper.pulses[0].amplitude

        return results
