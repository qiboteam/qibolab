import math
from qibo import K
from qibolab import states
from qibo.config import raise_error
from qibo.core import circuit

from qibolab.pulse import pulse


class PulseSequence:
    """List of pulses.

    Holds a separate list for each instrument.
    """

    def __init__(self):
        super().__init__()
        self.ro_pulses = []
        self.qd_pulses = []
        self.qf_pulses = []
        self.pulses = []
        self.time = 0
        self.phase = 0

    def __len__(self):
        return len(self.pulses)

    @property
    def serial(self):
        """Serial form of the whole sequence using the serial of each pulse."""
        return ", ".join(pulse.serial for pulse in self.pulses)

    def add(self, pulse):
        """Add a pulse to the sequence.

        Args:
            pulse (:class:`qibolab.pulses.Pulse`): Pulse object to add.

        Example:
            .. code-block:: python

                from qibolab.pulses import Pulse, ReadoutPulse, Rectangular, Gaussian, Drag
                from qibolab.circuit import PulseSequence
                # define two arbitrary pulses
                pulse1 = Pulse( start=0,
                                duration=60,
                                amplitude=0.3,
                                frequency=200_000_000.0,
                                phase=0,
                                shape=Gaussian(5),
                                channel=1,
                                type='qd')
                pulse2 = Pulse( start=70,
                                duration=2000,
                                amplitude=0.5,
                                frequency=20_000_000.0,
                                phase=0,
                                shape=Rectangular(),
                                channel=2,
                                type='ro')

                # define the pulse sequence
                sequence = PulseSequence()

                # add pulses to the pulse sequence
                sequence.add(pulse1)
                sequence.add(pulse2)
        """
        if pulse.type == "ro":
            self.ro_pulses.append(pulse)
        elif pulse.type == "qd":
            self.qd_pulses.append(pulse)
        elif pulse.type == "qf":
            self.qf_pulses.append(pulse)

        self.pulses.append(pulse)

    def add_u3(self, theta, phi, lam, qubit=0):
        """Add pulses that implement a U3 gate.
        Args:
            theta, phi, lam (float): Parameters of the U3 gate.
        """
        # apply RZ(lam)
        self.phase += lam
        # Fetch pi/2 pulse from calibration
        RX90_pulse_1= K.platform.RX90_pulse(qubit, self.time, self.phase)
        # apply RX(pi/2)
        self.add(RX90_pulse_1)
        self.time += RX90_pulse_1.duration
        # apply RZ(theta)
        self.phase += theta
        # Fetch pi/2 pulse from calibration
        RX90_pulse_2= K.platform.RX90_pulse(qubit, self.time, self.phase - math.pi)
        # apply RX(-pi/2)
        self.add(RX90_pulse_2)
        self.time += RX90_pulse_2.duration
        # apply RZ(phi)
        self.phase += phi

    def add_measurement(self, qubit=0):
        """Add measurement pulse."""
        MZ_pulse = K.platform.MZ_pulse(qubit, self.time, self.phase)
        self.add(MZ_pulse)
        self.time += MZ_pulse.duration


class HardwareCircuit(circuit.Circuit):

    def __init__(self, nqubits):
        if nqubits > 1: # TODO: Fetch platform nqubits
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

        # TODO: To be replaced with a proper classification of the states
        if K.platform.name == 'icarusq':
            q = self.measurement_gate.target_qubits[0]
            qubit = K.platform.fetch_qubit(q)
            min_v = qubit.min_readout_voltage
            max_v = qubit.max_readout_voltage
        else:
            qubit = self.measurement_gate.target_qubits[0]
            readout = list(list(readout.values())[0].values())[0]
            min_v = K.platform.settings['characterization']['single_qubit'][qubit]['rabi_oscillations_pi_pulse_min_voltage']
            max_v = K.platform.settings['characterization']['single_qubit'][qubit]['resonator_spectroscopy_max_ro_voltage']
        return states.HardwareState.from_readout(readout, min_v, max_v)
