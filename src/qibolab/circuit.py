import math
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
        self.pulses = []
        self.time = 0
        self.phase = 0        

    def __len__(self):
        return len(self.pulses)

    def serial(self):
        """Serial form of the whole sequence using the serial of each pulse."""
        return ", ".join(pulse.serial() for pulse in self.pulses)

    def add(self, pulse):
        """Add a pulse to the sequence.

        Args:
            pulse (:class:`qibolab.pulses.Pulse`): Pulse object to add.

        Example:
            .. code-block:: python

                from qibolab.pulses import Pulse, ReadoutPulse
                from qibolab.circuit import PulseSequence
                from qibolab.pulse_shapes import Rectangular, Gaussian

                # define two arbitrary pulses
                pulse1 = Pulse(start=0,
                               frequency=200000000.0,
                               amplitude=0.3,
                               duration=60,
                               phase=0,
                               shape=Gaussian(5)))
                pulse2 = ReadoutPulse(start=70,
                                      frequency=20000000.0,
                                      amplitude=0.5,
                                      duration=3000,
                                      phase=0,
                                      shape=Rectangular()))

                # define the pulse sequence
                sequence = PulseSequence()

                # add pulses to the pulse sequence
                sequence.add(pulse1)
                sequence.add(pulse2)
        """
        if pulse.channel == "qrm" or pulse.channel == 1:
            self.qrm_pulses.append(pulse)
        else:
            self.qcm_pulses.append(pulse)
        self.pulses.append(pulse)

    def add_u3(self, theta, phi, lam, qubit=0):
        """Add pulses that implement a U3 gate.

        Args:
            theta, phi, lam (float): Parameters of the U3 gate.
        """
        from qibolab.pulse_shapes import Gaussian
        # Fetch pi/2 pulse from calibration
        if hasattr(K.platform, "qubits"):
            kwargs = K.platform.fetch_qubit_pi_pulse(qubit)
        else:
            kwargs = {
                "amplitude": K.platform.pi_pulse_amplitude,
                "duration": K.platform.pi_pulse_duration,
                "frequency": K.platform.pi_pulse_frequency
            }
        kwargs["duration"] = kwargs["duration"] // 2
        delay = K.platform.delay_between_pulses
        duration = kwargs.get("duration")
        kwargs["shape"] = Gaussian(5)

        # apply RZ(lam)
        self.phase += lam
        # apply RX(pi/2)
        kwargs["start"] = self.time
        kwargs["phase"] = self.phase
        self.add(pulses.Pulse(**kwargs))
        self.time += duration + delay
        # apply RZ(theta)
        self.phase += theta
        # apply RX(-pi/2)
        kwargs["start"] = self.time
        kwargs["phase"] = self.phase - math.pi
        self.add(pulses.Pulse(**kwargs))
        self.time += duration + delay
        # apply RZ(phi)
        self.phase += phi

    def add_measurement(self, qubit=0):
        """Add measurement pulse."""
        from qibolab.pulse_shapes import Rectangular
        if hasattr(K.platform, "qubits"):
            kwargs = K.platform.fetch_qubit_readout_pulse(qubit)
        else:
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

    def create_sequence(self):
        """Creates the :class:`qibolab.circuit.PulseSequence` corresponding to the circuit's gates."""
        if self.measurement_gate is None:
            raise_error(RuntimeError, "No measurement register assigned.")

        sequence = PulseSequence()
        for gate in self.queue:
            gate.to_sequence(sequence)
        self.measurement_gate.to_sequence(sequence)
        return sequence

    def execute(self, initial_state=None, nshots=None):
        if initial_state is not None:
            raise_error(ValueError, "Hardware backend does not support "
                                    "initial state in circuits.")

        # Translate gates to pulses and create a ``PulseSequence``
        sequence = self.create_sequence()

        # Execute the pulse sequence on the platform
        K.platform.connect()
        K.platform.setup()
        K.platform.start()
        readout = K.platform(sequence, nshots)
        K.platform.stop()

        if hasattr(K.platform, "qubits"):
            q = self.measurement_gate.target_qubits[0]
            qubit = K.platform.fetch_qubit(q)
            min_v = qubit.min_readout_voltage
            max_v = qubit.max_readout_voltage
        else:
            min_v = K.platform.min_readout_voltage
            max_v = K.platform.max_readout_voltage
        return states.HardwareState.from_readout(readout, min_v, max_v)
