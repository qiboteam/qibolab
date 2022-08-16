import numpy as np
from qibo.config import raise_error


class U3Params:

    @staticmethod
    def H():
        return (7 * np.pi / 2, np.pi, 0)

    @staticmethod
    def X():
        return (np.pi, 0, np.pi)

    @staticmethod
    def Y():
        return (np.pi, 0, 0)

    @staticmethod
    def Z():
        return (0, np.pi, 0)

    @staticmethod
    def I(n):
        raise_error(NotImplementedError, "Identity gate is not implemented via U3.")

    @staticmethod
    def M():
        raise_error(NotImplementedError)

    @staticmethod
    def RX(theta):
        return (theta, -np.pi / 2, np.pi / 2)
        
    @staticmethod
    def RY(theta):
        return (theta, 0, 0)
        
    @staticmethod
    def RZ(theta):
        # apply virtually by changing ``phase`` instead of using pulses
        return (0, theta / 2, theta / 2)
        
    @staticmethod
    def U2(phi, lam):
        return (np.pi / 2, phi, lam)
        
    @staticmethod
    def U3(theta, phi, lam):
        return (theta, phi, lam)
        
    @staticmethod
    def Unitary(matrix):
        # https://github.com/Qiskit/qiskit-terra/blob/d2e3340adb79719f9154b665e8f6d8dc26b3e0aa/qiskit/quantum_info/synthesis/one_qubit_decompose.py#L221
        from scipy.linalg import det
        su2 = matrix / np.sqrt(det(matrix))
        theta = 2 * np.arctan2(abs(su2[1, 0]), abs(su2[0, 0]))
        plus = np.angle(su2[1, 1])
        minus = np.angle(su2[1, 0])
        phi = plus + minus
        lam = plus - minus
        return theta, phi, lam
