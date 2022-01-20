from qibo import K
from qibolab import states, pulses
from qibo.config import raise_error
from qibo.core import circuit


class PulseSequence:
    """List of pulses.

    Holds a separate list for each instrument.
    """

    def __init__(self):
        super().__init__()
        self.qcm_pulses = []
        self.qrm_pulses = []
        self.time = 0
        self.phase = 0

    def add(self, pulse):
        """Add a pulse to the sequence.

        Args:
            pulse (`qibolab.pulses.Pulse`): Pulse object to add.
            delay_between_pulses (int): Time to wait before applying the next pulse.
        """
        if pulse.channel == "qrm" or pulse.channel == 1:
            self.qrm_pulses.append(pulse)
        else:
            self.qcm_pulses.append(pulse)

    def add_u3(self, theta, phi, lam):
        """Add pulses that implement a U3 gate.

        Args:
            theta, phi, lam (float): Parameters of the U3 gate.
        """
        from qibolab.pulse_shapes import Gaussian
        # Pi/2 pulse from calibration
        amplitude = K.platform.pi_pulse_amplitude
        duration = K.platform.pi_pulse_duration // 2
        frequency = K.platform.pi_pulse_frequency
        delay = K.platform.delay_between_pulses

        self.phase += phi - np.pi / 2
        self.add(pulses.Pulse(self.time, duration, amplitude, frequency, self.phase, Gaussian(duration / 5)))
        self.time += duration + delay
        self.phase += np.pi - theta
        self.add(pulses.Pulse(self.time, duration, amplitude, frequency, self.phase, Gaussian(duration / 5)))
        self.time += duration + delay
        self.phase += lam - np.pi / 2

    def add_measurement(self):
        """Add measurement pulse."""
        from qibolab.pulse_shapes import Rectangular
        kwargs = K.platform.readout_pulse
        kwargs["start"] = self.time + K.platform.delay_before_readout
        kwargs["phase"] = self.phase
        kwargs["shape"] = Rectangular()
        self.add(pulses.ReadoutPulse(**kwargs))


class HardwareCircuit(circuit.Circuit):

    def __init__(self, nqubits):
        if nqubits > 1:
            raise ValueError("Device has only one qubit.")
        super().__init__(nqubits)

    def execute(self, initial_state=None, nshots=None):
        if initial_state is not None:
            raise_error(ValueError, "Hardware backend does not support "
                                    "initial state in circuits.")
        if self.measurement_gate is None:
            raise_error(RuntimeError, "No measurement register assigned.")

        # Translate gates to pulses and create a ``PulseSequence``
        sequence = PulseSequence()
        for gate in self.queue:
            gate.to_sequence(sequence)
        self.measurement_gate.to_sequence(sequence)

        # Execute the pulse sequence on the platform
        K.platform.start()
        readout = K.platform(sequence, nshots)
        K.platform.stop()

        min_v = K.platform.min_readout_voltage
        max_v = K.platform.max_readout_voltage
        return states.HardwareState.from_readout(readout, min_v, max_v)
