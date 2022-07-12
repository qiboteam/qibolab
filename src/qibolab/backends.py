import math
from qibo import gates
from qibo.config import raise_error
from qibo.states import CircuitResult
from qibo.backends import NumpyBackend


class U3Params:

    @property
    def H(self):
        return (7 * math.pi / 2, math.pi, 0)

    @property
    def X(self):
        return (math.pi, 0, math.pi)

    @property
    def Y(self):
        return (math.pi, 0, 0)

    @property
    def Z(self):
        return (0, math.pi, 0)

    def I(self, n):
        raise_error(NotImplementedError, "Identity gate is not implemented via U3.")

    @property
    def M(self):
        raise_error(NotImplementedError)

    def RX(self, theta):
        return (theta, -math.pi / 2, math.pi / 2)

    def RY(self, theta):
        return (theta, 0, 0)

    def RZ(self, theta):
        # apply virtually by changing ``phase`` instead of using pulses
        return (0, theta / 2, theta / 2)

    def U2(self, phi, lam):
        return (math.pi / 2, phi, lam)

    def U3(self, theta, phi, lam):
        return (theta, phi, lam)

    def Unitary(self, matrix):
        # https://github.com/Qiskit/qiskit-terra/blob/d2e3340adb79719f9154b665e8f6d8dc26b3e0aa/qiskit/quantum_info/synthesis/one_qubit_decompose.py#L221
        import numpy as np
        from scipy.linalg import det
        su2 = matrix / np.sqrt(det(matrix))
        theta = 2 * np.arctan2(abs(su2[1, 0]), abs(su2[0, 0]))
        plus = np.angle(su2[1, 1])
        minus = np.angle(su2[1, 0])
        phi = plus + minus
        lam = plus - minus
        return theta, phi, lam


class QibolabBackend(NumpyBackend):

    def __init__(self, platform, runcard=None):
        from qibolab.platform import Platform
        super().__init__()
        self.name = "qibolab"
        self.u3params = U3Params()
        self.platform = Platform(platform, runcard)

    def asu3(self, gate):
        name = gate.__class__.__name__
        if isinstance(gate, gates.ParametrizedGate):
            return getattr(self.u3params, name)(*gate.parameters)
        else:
            return getattr(self.u3params, name)

    def to_sequence(self, sequence, gate):
        if isinstance(gate, gates.M):
            # Add measurement pulse
            for qubit in gate.target_qubits:
                MZ_pulse = self.platform.MZ_pulse(qubit, sequence.time, sequence.phase)
                sequence.add(MZ_pulse)
                sequence.time += MZ_pulse.duration

        elif isinstance(gate, gates.I):
            pass

        elif isinstance(gate, gates.Z):
            sequence.phase += gate.parameters[0]

        else:
            if len(gate.qubits) > 1:
                raise_error(NotImplementedError, "Only one qubit gates are implemented.")

            qubit = gate.target_qubits[0]
            # Transform gate to U3 and add pi/2-pulses
            theta, phi, lam = self.asu3(gate)
            # apply RZ(lam)
            sequence.phase += lam
            # Fetch pi/2 pulse from calibration
            RX90_pulse_1 = self.platform.RX90_pulse(qubit, sequence.time, sequence.phase)
            # apply RX(pi/2)
            sequence.add(RX90_pulse_1)
            sequence.time += RX90_pulse_1.duration
            # apply RZ(theta)
            sequence.phase += theta
            # Fetch pi/2 pulse from calibration
            RX90_pulse_2 = self.platform.RX90_pulse(qubit, sequence.time, sequence.phase - math.pi)
            # apply RX(-pi/2)
            sequence.add(RX90_pulse_2)
            sequence.time += RX90_pulse_2.duration
            # apply RZ(phi)
            sequence.phase += phi

    def apply_gate(self, gate, state, nqubits): # pragma: no cover
        raise_error(NotImplementedError, "Qibolab cannot apply gates directly.")

    def apply_gate_density_matrix(self, gate, state, nqubits): # pragma: no cover
        raise_error(NotImplementedError, "Qibolab cannot apply gates directly.")

    def execute_circuit(self, circuit, initial_state=None, nshots=None): # pragma: no cover
        """Executes a quantum circuit.

        Args:
            circuit (:class:`qibo.core.circuit.Circuit`): Circuit to execute.
            nshots (int): Number of shots to sample from the experiment.
                If ``None`` the default value provided as hardware_avg in the
                calibration yml will be used.

        Returns:
            Readout results acquired by after execution.
        """
        from qibolab.pulses import PulseSequence
        if initial_state is not None:
            raise_error(ValueError, "Hardware backend does not support "
                                    "initial state in circuits.")

        # Translate gates to pulses and create a ``PulseSequence``
        if circuit.measurement_gate is None:
            raise_error(RuntimeError, "No measurement register assigned.")

        sequence = PulseSequence()
        for gate in circuit.queue:
            self.to_sequence(sequence, gate)
        self.to_sequence(sequence, circuit.measurement_gate)

        # Execute the pulse sequence on the platform
        self.platform.connect()
        self.platform.setup()
        self.platform.start()
        readout = self.platform(sequence, nshots)
        self.platform.stop()

        return CircuitResult(self, circuit, readout, nshots)

    def get_state_tensor(self):
        raise_error(NotImplementedError, "Qibolab cannot return state vector.")

    def get_state_repr(self, result): # pragma: no cover
        return result.execution_result
