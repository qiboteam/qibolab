import sys
import math
import copy
from abc import ABC, abstractmethod
from qibo import gates
from qibo.abstractions import abstract_gates
from qibo.config import raise_error


class AbstractHardwareGate(abstract_gates.Gate):
    module = sys.modules[__name__]

    @abstractmethod
    def to_u3_params(self): # pragma: no cover
        """Returns the angles of a U3 gate which implements the current gate."""
        raise_error(NotImplementedError)

    def to_sequence(self, sequence):
        """Adds the pulses implementing the gate to the given ``PulseSequence``."""
        args = self.to_u3_params() + (self.target_qubits[0],)
        sequence.add_u3(*args)


class H(AbstractHardwareGate, gates.H):

    def to_u3_params(self):
        return (7 * math.pi / 2, math.pi, 0)


class X(AbstractHardwareGate, gates.X):

    def to_u3_params(self):
        return (math.pi, 0, math.pi)


class Y(AbstractHardwareGate, gates.Y):

    def to_u3_params(self):
        return (math.pi, 0, 0)


class Z(AbstractHardwareGate, gates.Z):

    def to_u3_params(self):
        return (0, math.pi, 0)


class I(AbstractHardwareGate, gates.I):

    def to_u3_params(self):
        raise_error(NotImplementedError, "Identity gate is not implemented via U3.")

    def to_sequence(self, sequence):
        pass


class Align(AbstractHardwareGate, gates.I):  # pragma: no cover
    # TODO: Is this gate still needed?

    def to_sequence(self, sequence):
        raise_error(NotImplementedError)


class M(AbstractHardwareGate, gates.M):

    def to_u3_params(self):
        raise_error(NotImplementedError, "Measurement gate is not implemented via U3.")

    def to_sequence(self, sequence):
        for q in self.target_qubits:
            sequence.add_measurement(q)


class RX(AbstractHardwareGate, gates.RX):

    def to_u3_params(self):
        return (self.parameters, -math.pi / 2, math.pi / 2)


class RY(AbstractHardwareGate, gates.RY):

    def to_u3_params(self):
        return (self.parameters, 0, 0)


class RZ(AbstractHardwareGate, gates.RZ):

    def to_u3_params(self):
        return (0, self.parameters / 2, self.parameters / 2)

    def to_sequence(self, sequence):
        # apply virtually by changing ``phase`` instead of using pulses
        sequence.phase += self.parameters


class CNOT(AbstractHardwareGate, gates.CNOT):  # pragma: no cover

    def to_u3_params(self):
        raise_error(NotImplementedError)

    def to_sequence(self, sequence):
        raise_error(NotImplementedError)


class U2(AbstractHardwareGate, gates.U2):

    def to_u3_params(self):
        return (math.pi / 2,) + self.parameters


class U3(AbstractHardwareGate, gates.U3):

    def to_u3_params(self):
        return self.parameters
