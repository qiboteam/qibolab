import math
from qibo.config import raise_error


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
