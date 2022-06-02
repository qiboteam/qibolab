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

        if hasattr(K.platform, "qubits"):
            q = self.measurement_gate.target_qubits[0]
            qubit = K.platform.fetch_qubit(q)
            min_v = qubit.min_readout_voltage
            max_v = qubit.max_readout_voltage
        else:
            min_v = K.platform.min_readout_voltage
            max_v = K.platform.max_readout_voltage
        return states.HardwareState.from_readout(readout, min_v, max_v)
