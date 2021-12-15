import copy
import numpy as np
from qibo import gates
from qibo.config import raise_error
from qibo.core import measurements, circuit
from qibolab import tomography, experiment, scheduler, pulses, platform
from qibolab.gates import Align


class PulseSequence:
    """Describes a sequence of pulses for the FPGA to unpack and convert into arrays

    Current FPGA binary has variable sampling rate but fixed sample size.
    Due to software limitations we need to prepare all 16 DAC channel arrays.
    @see BasicPulse, MultifrequencyPulse and FilePulse for more information about supported pulses.

    Args:
        pulses: Array of Pulse objects
    """
    def __init__(self, pulses, duration=experiment.default_pulse_duration, sample_size=experiment.default_sample_size,
                 fixed_readout=True, time_offset=experiment.readout_pulse_duration() + 1e-6):
        if duration is None and sample_size is None:
            raise_error(RuntimeError("Either duration or sample size need to be defined"))

        self.pulses = pulses
        self.nchannels = experiment.nchannels
        self.sampling_rate = experiment.sampling_rate()
        if duration is not None:
            self.duration = duration
            self.sample_size = int(self.sampling_rate * duration)

        if sample_size is not None:
            self.sample_size = sample_size
            self.duration = sample_size / self.sampling_rate

            #if duration is not None:
                #warn("Defaulting to fixed sample size")
        #self.file_dir = experiment.pulse_file # NIU

        if fixed_readout:
            self.time = np.linspace(time_offset - self.duration, time_offset, num=self.sample_size)
        else:
            self.time = np.linspace(0, self.duration, self.sample_size)

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


class HardwareCircuit(circuit.Circuit):

    def __init__(self, nqubits, force_tomography=False):
        super().__init__(nqubits)
        self._force_tomography = force_tomography
        self._raw = None

    def _add_layer(self):
        raise_error(NotImplementedError, "VariationalLayer gate is not "
                                         "implemented for hardware backends.")

    def fuse(self):
        raise_error(NotImplementedError, "Circuit fusion is not implemented "
                                         "for hardware backends.")

    def _calculate_sequence_duration(self, gate_sequence):
        qubit_times = np.zeros(self.nqubits)
        for gate in gate_sequence:
            q = gate.target_qubits[0]


            if isinstance(gate, Align):
                m = 0
                for q in gate.target_qubits:
                    m = max(m, qubit_times[q])

                for q in gate.target_qubits:
                    qubit_times[q] = m

            # TODO: Condition for two/three qubit gates
            elif isinstance(gate, gates.CNOT):
                # CNOT cannot be tested because calibration placeholder supports single qubit only
                control = gate.control_qubits[0]
                start = max(qubit_times[q], qubit_times[control])
                qubit_times[q] = start + gate.duration(experiment.qubits)
                qubit_times[control] = qubit_times[q]

            else:
                qubit_times[q] += gate.duration(experiment.qubits)

        return qubit_times

    def create_pulse_sequence(self, queue, qubit_times, qubit_phases):
        args = [experiment.qubits, qubit_times, qubit_phases]
        sequence = []
        for gate in queue:
            sequence.extend(gate.pulse_sequence(*args))
        sequence.extend(self.measurement_gate.pulse_sequence(*args))
        return PulseSequence(sequence)

    def _execute_sequence(self, nshots):
        """For one qubit, we can rely on IQ data projection to get the probability p."""
        target_qubits = self.measurement_gate.target_qubits
        # Calculate qubit control pulse duration and move it before readout
        qubit_phases = np.zeros(self.nqubits)
        qubit_times = experiment.static.readout_start_time - self._calculate_sequence_duration(self.queue)
        pulse_sequence = self.create_pulse_sequence(self.queue, qubit_times, qubit_phases)
        # Execute pulse sequence and project data to probability if requested
        self._raw = scheduler.execute_pulse_sequence(pulse_sequence, nshots)
        probabilities = self._raw[0]
        # TODO: Adjust for single shot measurements
        output = measurements.MeasurementResult(target_qubits, probabilities, nshots)
        self._final_state = output

        return self._final_state

    def _execute_tomography_sequence(self, nshots):
        """For 2+ qubits, since we do not have a TWPA
        (and individual qubit resonator) on this system,
        we need to do tomography to get the density matrix
        """
        # TODO: n-qubit dynamic tomography
        target_qubits = self.measurement_gate.target_qubits
        nqubits = len(target_qubits)
        ps_states = tomography.Tomography.basis_states(nqubits)
        prerotation = tomography.Tomography.gate_sequence(nqubits)
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

        self._raw = scheduler.execute_tomography(ps_array, nshots)
        density_matrix = self._raw[0]
        probabilities = np.array([density_matrix[k, k].real for k in range(nqubits)])
        target_qubits = self.measurement_gate.target_qubits
        output = measurements.MeasurementResult(target_qubits, probabilities, nshots)
        self._final_state = output

        return self._final_state

    def execute(self, initial_state=None, nshots=None):
        if initial_state is not None:
            raise_error(ValueError, "Hardware backend does not support "
                                    "initial state in circuits.")

        if self.measurement_gate is None:
            raise_error(RuntimeError, "No measurement register assigned")

        # Get calibration data
        self.qubit_config = scheduler.fetch_config()

        # TODO: Move data fitting to qibolab.experiments
        if len(scheduler.check_tomography_required() or self._force_tomography):
            return self._execute_tomography_sequence(nshots)
        else:
            return self._execute_sequence(nshots)

    def __call__(self, initial_state=None, nshots=None):
        return self.execute(initial_state, nshots)


class TIICircuit(circuit.Circuit):

    def __init__(self, nqubits):
        if nqubits > 1:
            raise ValueError("Device has only one qubit.")
        super().__init__(nqubits)

    def execute(self, initial_state=None, nshots=None):
        if initial_state is not None:
            raise_error(ValueError, "Hardware backend does not support "
                                    "initial state in circuits.")

        sequence = pulses.PulseSequence()
        for gate in self.queue:
            sequence.add_u3(*gate.to_u3_parameters())

        if self.measurement_gate is not None:
            sequence.add_measurement()
        else:
            raise_error(RuntimeError, "No measurement register assigned.")

        platform.start()
        result = platform(sequence, nshots)
        platform.stop()
        return result
