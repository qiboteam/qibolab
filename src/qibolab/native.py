# -*- coding: utf-8 -*-
import numpy as np
from qibo import gates
from qibo.config import raise_error

from qibolab.transpilers.decompositions import two_qubit_decomposition, u3_decomposition


class NativeGates:
    """Maps Qibo gates to a hardware native implementation."""

    def __init__(self, backend):
        self.backend = backend

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
        raise_error(NotImplementedError)

    def SDG(self, gate):
        raise_error(NotImplementedError)

    def T(self, gate):
        raise_error(NotImplementedError)

    def TDG(self, gate):
        raise_error(NotImplementedError)

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
        matrix = self.backend.asmatrix(gate)
        return two_qubit_decomposition(q0, q1, matrix)

    def CRY(self, gate):
        q0, q1 = gate.qubits
        matrix = self.backend.asmatrix(gate)
        return two_qubit_decomposition(q0, q1, matrix)

    def CRZ(self, gate):
        q0, q1 = gate.qubits
        matrix = self.backend.asmatrix(gate)
        return two_qubit_decomposition(q0, q1, matrix)

    def CU1(self, gate):
        q0, q1 = gate.qubits
        matrix = self.backend.asmatrix(gate)
        return two_qubit_decomposition(q0, q1, matrix)

    def CU2(self, gate):
        q0, q1 = gate.qubits
        matrix = self.backend.asmatrix(gate)
        return two_qubit_decomposition(q0, q1, matrix)

    def CU3(self, gate):
        q0, q1 = gate.qubits
        matrix = self.backend.asmatrix(gate)
        return two_qubit_decomposition(q0, q1, matrix)

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
        matrix = self.backend.asmatrix(gate)
        return two_qubit_decomposition(q0, q1, matrix)

    def fSim(self, gate):
        q0, q1 = gate.qubits
        matrix = self.backend.asmatrix(gate)
        return two_qubit_decomposition(q0, q1, matrix)

    def GeneralizedfSim(self, gate):
        q0, q1 = gate.qubits
        matrix = self.backend.asmatrix(gate)
        return two_qubit_decomposition(q0, q1, matrix)

    def RXX(self, gate):
        q0, q1 = gate.qubits
        matrix = self.backend.asmatrix(gate)
        return two_qubit_decomposition(q0, q1, matrix)

    def RYY(self, gate):
        q0, q1 = gate.qubits
        matrix = self.backend.asmatrix(gate)
        return two_qubit_decomposition(q0, q1, matrix)

    def RZZ(self, gate):
        q0, q1 = gate.qubits
        matrix = self.backend.asmatrix(gate)
        return two_qubit_decomposition(q0, q1, matrix)

    def TOFFOLI(self, gate):
        raise_error(NotImplementedError)

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
        raise_error(NotImplementedError)
