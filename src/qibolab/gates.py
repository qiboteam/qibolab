import sys
import math
import copy
from abc import ABC, abstractmethod
from qibo import gates
from qibo.config import raise_error


class AbstractHardwareGate(ABC):
    module = sys.modules[__name__]

    @abstractmethod
    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases): # pragma: no cover
        raise_error(NotImplementedError)

    @abstractmethod
    def duration(self, qubit_config): # pragma: no cover
        raise_error(NotImplementedError)

    @abstractmethod
    def to_u3_params(self): # pragma: no cover
        """Returns the angles of a U3 gate which implements the current gate."""
        raise_error(NotImplementedError)

    def to_sequence(self, sequence):
        """Adds the pulses implementing the gate to the given ``PulseSequence``."""
        args = self.to_u3_params() + (self.target_qubits[0],)
        sequence.add_u3(*args)


class H(AbstractHardwareGate, gates.H):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        q = self.target_qubits[0]
        composite = [RY(q, math.pi / 2), RX(q, math.pi)]
        pulses = []
        for gate in composite:
            pulses.extend(gate.pulse_sequence(qubit_config, qubit_times, qubit_phases))
        return pulses

    def duration(self, qubit_config):
        d = 0
        q = self.target_qubits[0]
        composite = [RY(q, math.pi / 2), RX(q, math.pi)]
        for gate in composite:
            d += gate.duration(qubit_config)
        return d

    def to_u3_params(self):
        return (7 * math.pi / 2, math.pi, 0)


class I(AbstractHardwareGate, gates.I):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        return []

    def duration(self, qubit_config):
        return 0

    def to_u3_params(self):  # pragma: no cover
        raise_error(NotImplementedError, "Identity gate is not implemented via U3.")

    def to_sequence(self, sequence):
        pass


class Align(AbstractHardwareGate, gates.I):  # pragma: no cover
    # TODO: Is this gate still needed?

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        m = max(qubit_times[q] for q in self.target_qubits)
        for q in self.target_qubits:
            qubit_times[q] = m
        return []

    def duration(self, qubit_config):
        return 0

    def to_sequence(self, sequence):
        raise_error(NotImplementedError)


class M(AbstractHardwareGate, gates.M):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        pulses = []
        for q in self.target_qubits:
            pulses += copy.deepcopy(qubit_config[q].gates.get(self))
        return pulses

    def duration(self, qubit_config):
        pulses = []
        for q in self.target_qubits:
            pulses += copy.deepcopy(qubit_config[q].gates.get(self))
        m = 0
        for p in pulses:
            m = max(p.duration, m)
        return m

    def to_u3_params(self):
        raise_error(NotImplementedError, "Measurement gate is not implemented via U3.")

    def to_sequence(self, sequence):
        for q in self.target_qubits:
            sequence.add_measurement(q)


class RX(AbstractHardwareGate, gates.RX):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        if self.parameters == 0:
            return []

        q = self.target_qubits[0]
        time_mod = abs(self.parameters / math.pi)
        phase_mod = 0 if self.parameters > 0 else -180
        phase_mod += qubit_phases[q]
        m = 0

        pulses = copy.deepcopy(qubit_config[q].gates.get(self))
        for p in pulses:
            duration = p.duration * time_mod
            p.start = qubit_times[q]
            p.phase += phase_mod
            p.duration = duration
            m = max(duration, m)
        qubit_times[q] += m

        return pulses

    def duration(self, qubit_config):
        q = self.target_qubits[0]
        time_mod = abs(self.parameters / math.pi)
        pulses = copy.deepcopy(qubit_config[q].gates.get(self))
        m = 0

        for p in pulses:
            m = max(p.duration * time_mod, m)
        return m

    def to_u3_params(self):
        return (self.parameters, -math.pi / 2, math.pi / 2)


class RY(AbstractHardwareGate, gates.RY):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        return RX.pulse_sequence(self, qubit_config, qubit_times, qubit_phases)

    def duration(self, qubit_config):
        return RX.duration(self, qubit_config)

    def to_u3_params(self):
        return (self.parameters, 0, 0)


class RZ(AbstractHardwareGate, gates.RZ):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases): # pragma: no cover
        raise_error(NotImplementedError)

    def duration(self, qubit_config): # pragma: no cover
        raise_error(NotImplementedError)

    def to_u3_params(self):
        return (0, self.parameters / 2, self.parameters / 2)

    def to_sequence(self, sequence):
        # apply virtually by changing ``phase`` instead of using pulses
        sequence.phase += self.parameters


class CNOT(AbstractHardwareGate, gates.CNOT):
    # CNOT gate is not tested because `qubit_config` placeholder is single qubit

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        q = self.target_qubits[0]
        control = self.control_qubits[0]
        start = max(qubit_times[q], qubit_times[control])
        pulses = copy.deepcopy(qubit_config[q].gates.get(self))

        for p in pulses:
            duration = p.duration
            p.start = start
            p.phase = qubit_phases[q]
            p.duration = duration
            qubit_times[q] = start + duration

        qubit_times[control] = qubit_times[q]
        return pulses

    def duration(self, qubit_config):
        q = self.target_qubits[0]
        control = self.control_qubits[0]
        m = 0
        pulses = qubit_config[q]["gates"][self.name + "_{}".format(control)]

        for p in pulses:
            m = max(p.duration, m)
        return m

    def to_u3_params(self):
        raise_error(NotImplementedError)

    def to_sequence(self, sequence):
        raise_error(NotImplementedError)


class U2(AbstractHardwareGate, gates.U2):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases): # pragma: no cover
        raise_error(NotImplementedError)

    def duration(self, qubit_config): # pragma: no cover
        raise_error(NotImplementedError)

    def to_u3_params(self):
        return (math.pi / 2,) + self.parameters


class U3(AbstractHardwareGate, gates.U3):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases): # pragma: no cover
        raise_error(NotImplementedError)

    def duration(self, qubit_config): # pragma: no cover
        raise_error(NotImplementedError)

    def to_u3_params(self):
        return self.parameters


class X(AbstractHardwareGate, gates.X):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases): # pragma: no cover
        raise_error(NotImplementedError)

    def duration(self, qubit_config): # pragma: no cover
        raise_error(NotImplementedError)

    def to_u3_params(self):
        return (math.pi, 0, math.pi)


class Y(AbstractHardwareGate, gates.Y):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases): # pragma: no cover
        raise_error(NotImplementedError)

    def duration(self, qubit_config): # pragma: no cover
        raise_error(NotImplementedError)

    def to_u3_params(self):
        return (math.pi, 0, 0)


class Z(AbstractHardwareGate, gates.Z):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases): # pragma: no cover
        raise_error(NotImplementedError)

    def duration(self, qubit_config): # pragma: no cover
        raise_error(NotImplementedError)

    def to_u3_params(self):
        return (0, math.pi, 0)
