import copy
import numpy as np
from qibo import gates
from qibo.config import raise_error
from qibo.core import measurements, circuit
from qiboicarusq import pulses, tomography, experiment, scheduler


class PulseSequence:
    """Describes a sequence of pulses for the FPGA to unpack and convert into arrays

    Current FPGA binary has variable sampling rate but fixed sample size.
    Due to software limitations we need to prepare all 16 DAC channel arrays.
    @see BasicPulse, MultifrequencyPulse and FilePulse for more information about supported pulses.

    Args:
        pulses: Array of Pulse objects
    """
    def __init__(self, pulses, duration=None):
        self.pulses = pulses
        self.nchannels = experiment.static.nchannels
        self.sample_size = experiment.static.sample_size
        self.sampling_rate = experiment.static.sampling_rate
        self.file_dir = experiment.static.pulse_file

        if duration is None:
            self.duration = self.sample_size / self.sampling_rate
        else:
            self.duration = duration
            self.sample_size = int(duration * self.sampling_rate)
        end = experiment.static.readout_pulse_duration + 1e-6
        self.time = np.linspace(end - self.duration, end, num=self.sample_size)

    def compile(self):
        """Compiles pulse sequence into waveform arrays

        FPGA binary is currently unable to parse pulse sequences, so this is a temporary workaround to prepare the arrays

        Returns:
            Numpy.ndarray holding waveforms for each channel. Has shape (nchannels, sample_size).
        """
        waveform = np.zeros((self.nchannels, self.sample_size))
        for pulse in self.pulses:
            waveform = pulse.compile(waveform, self)
        return waveform

    def serialize(self):
        """Returns the serialized pulse sequence."""
        return ", ".join([pulse.serial() for pulse in self.pulses])


class HardwareCircuit(circuit.HardwareCircuit):

    @staticmethod
    def _probability_extraction(data, refer_0, refer_1):
        move = copy.copy(refer_0)
        refer_0 = refer_0 - move
        refer_1 = refer_1 - move
        data = data - move
        # Rotate the data so that vector 0-1 is overlapping with Ox
        angle = copy.copy(np.arccos(refer_1[0]/np.sqrt(refer_1[0]**2 + refer_1[1]**2))*np.sign(refer_1[1]))
        new_data = np.array([data[0]*np.cos(angle) + data[1]*np.sin(angle),
                             -data[0]*np.sin(angle) + data[1]*np.cos(angle)])
        # Rotate refer_1 to get state 1 reference
        new_refer_1 = np.array([refer_1[0]*np.cos(angle) + refer_1[1]*np.sin(angle),
                                -refer_1[0]*np.sin(angle) + refer_1[1]*np.cos(angle)])
        # Condition for data outside bound
        if new_data[0] < 0:
            new_data[0] = 0
        elif new_data[0] > new_refer_1[0]:
            new_data[0] = new_refer_1[0]
        return new_data[0] / new_refer_1[0]

    def _calculate_sequence_duration(self, gate_sequence):
        qubit_times = np.zeros(self.nqubits)
        for gate in gate_sequence:
            q = gate.target_qubits[0]

            if isinstance(gate, gates.Align):
                m = 0
                for q in gate.target_qubits:
                    m = max(m, qubit_times[q])

                for q in gate.target_qubits:
                    qubit_times[q] = m

            elif isinstance(gate, gates.CNOT):
                control = gate.control_qubits[0]
                start = max(qubit_times[q], qubit_times[control])
                qubit_times[q] = start + gate.duration(self.qubit_config)
                qubit_times[control] = qubit_times[q]

            else:
                qubit_times[q] += gate.duration(self.qubit_config)

        return qubit_times

    def create_pulse_sequence(self, queue, qubit_times, qubit_phases):
        args = [self.qubit_config, qubit_times, qubit_phases]
        sequence = []
        for gate in queue:
            sequence.extend(gate.pulse_sequence(*args))
        sequence.extend(self.measurement_gate.pulse_sequence(*args))
        return PulseSequence(sequence)

    def _execute_one_qubit(self, nshots, measurement_level=2):
        """For one qubit, we can rely on IQ data projection to get the probability p."""
        target_qubits = self.measurement_gate.target_qubits
        # Calculate qubit control pulse duration and move it before readout
        qubit_phases = np.zeros(self.nqubits)
        qubit_times = experiment.static.readout_start_time - self._calculate_sequence_duration(self.queue)
        pulse_sequence = self.create_pulse_sequence(self.queue, qubit_times, qubit_phases)
        # Execute pulse sequence and project data to probability if requested
        job = scheduler.execute_pulse_sequence(pulse_sequence, nshots)
        raw_data = job.result()
        if measurement_level == 0:
            self._final_state = raw_data
        else:
            for q in target_qubits:
                data = self._parse_result(q, raw_data) # Get IQ values

                if measurement_level == 1:
                    output = data

                elif measurement_level == 2:
                    ref_zero = np.array(self.qubit_config[q]["iq_state"]["0"])
                    ref_one = np.array(self.qubit_config[q]["iq_state"]["1"])
                    p = self._probability_extraction(data, ref_zero, ref_one)
                    probabilities = np.array([1 - p, p])
                    output = measurements.MeasurementResult(target_qubits, probabilities, nshots)

            self._final_state = output

        return self._final_state

    def _execute_many_qubits(self, nshots, measurement_level=2):
        """For 2+ qubits, since we do not have a TWPA
        (and individual qubit resonator) on this system,
        we need to do tomography to get the density matrix
        """
        # TODO: n-qubit dynamic tomography
        ps_states = tomography.Tomography.basis_states(2)
        prerotation = tomography.Tomography.gate_sequence(2)
        ps_array = []
        # Set pulse sequence to get the state vectors
        for state_gate in ps_states:
            qubit_times = np.zeros(self.nqubits) - max(self._calculate_sequence_duration(state_gate))
            qubit_phases = np.zeros(self.nqubits)
            ps_array.append(self.create_pulse_sequence(state_gate, qubit_times, qubit_phases))

        # Append prerotation to the circuit sequence for tomography
        for prerotation_sequence in prerotation:
            qubit_phases = np.zeros(self.nqubits)
            seq = self.queue + [gates.Align(*tuple(range(self.nqubits)))] + prerotation_sequence
            qubit_times = np.zeros(self.nqubits) - max(self._calculate_sequence_duration(seq))
            ps_array.append(self.create_pulse_sequence(seq, qubit_times, qubit_phases))

        job = scheduler.execute_batch_sequence(ps_array, nshots)
        raw_data = job.result()

        if measurement_level == 0:
            self._final_state = raw_data
        else:
            data = [self._parse_result(0, raw_data[k]) for k in range(len(ps_array))] # Get IQ values

            if measurement_level == 1:
                output = data

            elif measurement_level == 2:
                # Map each measurement to the phase value and seperate state from prerotations
                data = np.array([np.arctan2(data[k][1], data[k][0]) * 180 / np.pi for k in range(len(ps_array))])
                states = data[0:4]
                amp = data[4:20]

                # TODO: Repeated minimize, tomography fail decision, tolerance
                tom = tomography.Tomography(amp, states)
                tom.minimize(1e-5)
                fit = tom.fit
                probabilities = np.array([fit[k, k].real for k in range(4)])
                target_qubits = self.measurement_gate.target_qubits
                output = measurements.MeasurementResult(target_qubits, probabilities, nshots)

            self._final_state = output

        return self._final_state

    def execute(self, nshots, measurement_level=2):
        super().execute(None, nshots, measurement_level)
        # Get calibration data
        self.qubit_config = scheduler.fetch_config()

        # Compile pulse sequence
        if self.measurement_gate is None:
            raise_error(RuntimeError, "No measurement register assigned")

        # Parse results according to desired measurement level

        # We use the same standard as OpenPulses measurement output level
        # Level 0: Raw
        # Level 1: IQ Values
        # Level 2: Qubit State

        # TODO: Move data fitting to qiboicarusq.experiments
        if len(self.measurement_gate.target_qubits) == 1:
            return self._execute_one_qubit(nshots, measurement_level)
        else:
            return self._execute_many_qubits(nshots, measurement_level)

    def _parse_result(self, qubit, raw_data):
        final = experiment.static.ADC_length / experiment.static.ADC_sampling_rate
        step = 1 / experiment.static.ADC_sampling_rate
        ADC_time_array = np.arange(0, final, step)[50:]

        static_data = experiment.static.qubit_static_parameters[self.qubit_config[qubit]["id"]]
        ro_channel = static_data["channel"][2]
        # For now readout is done with mixers
        IF_frequency = static_data["resonator_frequency"] - experiment.static.lo_frequency # downconversion

        #raw_data = self.final_state.result()
        # TODO: Implement a method to detect when the readout signal starts in the ADC data
        cos = np.cos(2 * np.pi * IF_frequency * ADC_time_array)
        it = np.sum(raw_data[ro_channel[0]] * cos)
        qt = np.sum(raw_data[ro_channel[1]] * cos)
        return np.array([it, qt])
