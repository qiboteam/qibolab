import numpy as np
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.config import raise_error

from qibolab.transpilers.decompositions import two_qubit_decomposition, u3_decomposition


class NativeGates:
    """Maps Qibo gates to a hardware native implementation."""

    def __init__(self):
        self.backend = NumpyBackend()

    def translate_gate(self, gate):
        name = gate.__class__.__name__
        return getattr(self, name)(gate)

    def translate_circuit(self, circuit, translate_single_qubit=False):
        """Translates a circuit to native gates."""
        new = circuit.__class__(circuit.nqubits)
        for gate in circuit.queue:
            if len(gate.qubits) > 1 or translate_single_qubit:
                new.add(self.translate_gate(gate))
            else:
                new.add(gate)
        return new

    def H(self, gate):
        q = gate.target_qubits[0]
        return gates.U3(q, 7 * np.pi / 2, np.pi, 0)

    def X(self, gate):
        q = gate.target_qubits[0]
        return gates.U3(q, np.pi, 0, np.pi)

    def Y(self, gate):
        q = gate.target_qubits[0]
        return gates.U3(q, np.pi, 0, 0)

    def Z(self, gate):
        # apply virtually by changing ``phase`` instead of using pulses
        return gate

    def S(self, gate):
        q = gate.target_qubits[0]
        return gates.RZ(q, np.pi / 2)

    def SDG(self, gate):
        q = gate.target_qubits[0]
        return gates.RZ(q, -np.pi / 2)

    def T(self, gate):
        q = gate.target_qubits[0]
        return gates.RZ(q, np.pi / 4)

    def TDG(self, gate):
        q = gate.target_qubits[0]
        return gates.RZ(q, -np.pi / 4)

    def I(self, gate):
        return gate

    def Align(self, gate):
        return gate

    def M(self, gate):
        raise_error(NotImplementedError)

    def RX(self, gate):
        q = gate.target_qubits[0]
        theta = gate.parameters[0]
        return gates.U3(q, theta, -np.pi / 2, np.pi / 2)

    def RY(self, gate):
        q = gate.target_qubits[0]
        theta = gate.parameters[0]
        return gates.U3(q, theta, 0, 0)

    def RZ(self, gate):
        # apply virtually by changing ``phase`` instead of using pulses
        return gate

    def U1(self, gate):
        # apply virtually by changing ``phase`` instead of using pulses
        q = gate.target_qubits[0]
        theta = gate.parameters[0]
        return gates.RZ(q, theta)

    def U2(self, gate):
        q = gate.target_qubits[0]
        phi, lam = gate.parameters
        return gates.U3(q, np.pi / 2, phi, lam)

    def U3(self, gate):
        return gate

    def CNOT(self, gate):
        q0, q1 = gate.qubits
        return [gates.H(q1), gates.CZ(q0, q1), gates.H(q1)]

    def CZ(self, gate):
        return gate

    def CRX(self, gate):
        q0, q1 = gate.qubits
        theta = gate.parameters[0] / 2.0
        return [
            gates.RX(q1, theta=theta),
            gates.CZ(q0, q1),
            gates.RX(q1, theta=-theta),
            gates.CZ(q0, q1),
        ]

    def CRY(self, gate):
        q0, q1 = gate.qubits
        theta = gate.parameters[0] / 2.0
        return [
            gates.RY(q1, theta=theta),
            gates.CZ(q0, q1),
            gates.RY(q1, theta=-theta),
            gates.CZ(q0, q1),
        ]

    def CRZ(self, gate):
        q0, q1 = gate.qubits
        theta = gate.parameters[0] / 2.0
        return [
            gates.RZ(q1, theta=theta),
            gates.H(q1),
            gates.CZ(q0, q1),
            gates.RX(q1, theta=-theta),
            gates.CZ(q0, q1),
            gates.H(q1),
        ]

    def CU1(self, gate):
        q0, q1 = gate.qubits
        theta = gate.parameters[0] / 2.0
        return [
            gates.RZ(q0, theta=theta),
            gates.H(q1),
            gates.CZ(q0, q1),
            gates.RX(q1, theta=-theta),
            gates.CZ(q0, q1),
            gates.H(q1),
            gates.RZ(q1, theta=theta),
        ]

    def CU2(self, gate):
        q0, q1 = gate.qubits
        phi, lam = gate.parameters
        q0, q1 = gate.qubits
        phi, lam = gate.parameters
        return [
            gates.RZ(q1, theta=(lam - phi) / 2.0),
            gates.H(q1),
            gates.CZ(q0, q1),
            gates.H(q1),
            gates.U3(q1, -np.pi / 4, 0, -(lam + phi) / 2.0),
            gates.H(q1),
            gates.CZ(q0, q1),
            gates.H(q1),
            gates.U3(q1, np.pi / 4, phi, 0),
        ]

    def CU3(self, gate):
        q0, q1 = gate.qubits
        theta, phi, lam = gate.parameters
        return [
            gates.RZ(q1, theta=(lam - phi) / 2.0),
            gates.H(q1),
            gates.CZ(q0, q1),
            gates.H(q1),
            gates.U3(q1, -theta / 2.0, 0, -(lam + phi) / 2.0),
            gates.H(q1),
            gates.CZ(q0, q1),
            gates.H(q1),
            gates.U3(q1, theta / 2.0, phi, 0),
        ]

    def SWAP(self, gate):
        q0, q1 = gate.qubits
        return [
            gates.H(q1),
            gates.CZ(q0, q1),
            gates.H(q1),
            gates.H(q0),
            gates.CZ(q1, q0),
            gates.H(q0),
            gates.H(q1),
            gates.CZ(q0, q1),
            gates.H(q1),
        ]

    def FSWAP(self, gate):
        q0, q1 = gate.qubits
        return [
            gates.U3(q0, np.pi / 2, -np.pi / 2, -np.pi),
            gates.U3(q1, np.pi / 2, np.pi / 2, np.pi / 2),
            gates.CZ(q0, q1),
            gates.U3(q0, np.pi / 2, 0, -np.pi / 2),
            gates.U3(q1, np.pi / 2, 0, np.pi / 2),
            gates.CZ(q0, q1),
            gates.U3(q0, np.pi / 2, np.pi / 2, -np.pi),
            gates.U3(q1, np.pi / 2, 0, -np.pi),
        ]

    def fSim(self, gate):
        q0, q1 = gate.qubits
        matrix = gate.asmatrix(self.backend)
        return two_qubit_decomposition(q0, q1, matrix)

    def GeneralizedfSim(self, gate):
        q0, q1 = gate.qubits
        matrix = gate.asmatrix(self.backend)
        return two_qubit_decomposition(q0, q1, matrix)

    def RXX(self, gate):
        q0, q1 = gate.qubits
        theta = gate.parameters
        return [
            gates.H(q0),
            gates.CZ(q0, q1),
            gates.RX(q1, theta),
            gates.CZ(q0, q1),
            gates.H(q0),
        ]

    def RYY(self, gate):
        q0, q1 = gate.qubits
        theta = gate.parameters
        return [
            gates.RX(q0, np.pi / 2),
            gates.U3(q1, np.pi / 2, np.pi / 2, -np.pi),
            gates.CZ(q0, q1),
            gates.RX(q1, theta),
            gates.CZ(q0, q1),
            gates.RX(q0, -np.pi / 2),
            gates.U3(q1, np.pi / 2, 0, np.pi / 2),
        ]

    def RZZ(self, gate):
        q0, q1 = gate.qubits
        theta = gate.parameters
        return [
            gates.H(q1),
            gates.CZ(q0, q1),
            gates.RX(q1, theta),
            gates.CZ(q0, q1),
            gates.H(q1),
        ]

    def TOFFOLI(self, gate):
        q0, q1, q2 = gate.qubits
        return [
            gates.CZ(q1, q2),
            gates.RX(q2, -np.pi / 4),
            gates.CZ(q0, q2),
            gates.RX(q2, np.pi / 4),
            gates.CZ(q1, q2),
            gates.RX(q2, -np.pi / 4),
            gates.CZ(q0, q2),
            gates.RX(q2, np.pi / 4),
            gates.RZ(q1, np.pi / 4),
            gates.H(q1),
            gates.CZ(q0, q1),
            gates.RZ(q0, np.pi / 4),
            gates.RX(q1, -np.pi / 4),
            gates.CZ(q0, q1),
            gates.H(q1),
        ]

    def Unitary(self, gate):
        matrix = gate.parameters[0]
        if len(gate.qubits) == 1:
            q = gate.qubits[0]
            theta, phi, lam = u3_decomposition(matrix)
            return gates.U3(q, theta, phi, lam)

        elif len(gate.qubits) == 2:
            q0, q1 = gate.qubits
            return two_qubit_decomposition(q0, q1, matrix)

        else:
            raise_error(
                NotImplementedError,
                "Cannot implement gates acting on more than two qubits.",
            )

    def CallbackGate(self, gate):  # pragma: no cover
        raise_error(NotImplementedError)

    def PartialTrace(self, gate):  # pragma: no cover
        raise_error(NotImplementedError)

    def UnitaryChannel(self, gate):  # pragma: no cover
        raise_error(NotImplementedError)

    def PauliNoiseChannel(self, gate):  # pragma: no cover
        raise_error(NotImplementedError)

    def ResetChannel(self, gate):  # pragma: no cover
        raise_error(NotImplementedError)

    def ThermalRelaxationChannel(self, gate):  # pragma: no cover
        raise_error(NotImplementedError)

    def FusedGate(self, gate):  # pragma: no cover
        matrix = gate.asmatrix(self.backend)
        fgate = gates.Unitary(matrix, *gate.qubits)
        return self.Unitary(fgate)
