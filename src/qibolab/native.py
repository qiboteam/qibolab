# -*- coding: utf-8 -*-
import numpy as np
from qibo import gates
from qibo.config import raise_error


class NativeGates:
    """Maps Qibo gates to a hardware native implementation."""

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

    def I(self, gate):
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

    def U2(self, gate):
        q = gate.target_qubits[0]
        phi, lam = gate.parameters
        return gates.U3(q, np.pi / 2, phi, lam)

    def U3(self, gate):
        return gate

    def Unitary(self, gate):
        if len(gate.target_qubits) > 1:
            raise_error(NotImplementedError)

        # https://github.com/Qiskit/qiskit-terra/blob/d2e3340adb79719f9154b665e8f6d8dc26b3e0aa/qiskit/quantum_info/synthesis/one_qubit_decompose.py#L221
        from scipy.linalg import det

        q = gate.target_qubits[0]
        matrix = gate.parameters[0]
        su2 = matrix / np.sqrt(det(matrix))
        theta = 2 * np.arctan2(abs(su2[1, 0]), abs(su2[0, 0]))
        plus = np.angle(su2[1, 1])
        minus = np.angle(su2[1, 0])
        phi = plus + minus
        lam = plus - minus
        return gates.U3(q, theta, phi, lam)
