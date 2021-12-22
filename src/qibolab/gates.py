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
    def to_sequence(self, sequence):  # pragma: no cover
        """Adds the pulses implementing the gate to the given ``PulseSequence``."""
        raise_error(NotImplementedError)


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

    def to_sequence(self, sequence):
        sequence.add_u3(7 * math.pi / 2, math.pi, 0)


class I(AbstractHardwareGate, gates.I):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        return []

    def duration(self, qubit_config):
        return 0

    def to_sequence(self, sequence):
        pass


class Align(AbstractHardwareGate, gates.I):

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

    def to_sequence(self, sequence):
        sequence.add_measurement()


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

    def to_sequence(self, sequence):
        q = self.target_qubits[0]
        theta = self.parameters
        phi = - math.pi / 2
        lam = math.pi / 2
        sequence.add_u3(theta, phi, lam)


class RY(AbstractHardwareGate, gates.RY):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        return RX.pulse_sequence(self, qubit_config, qubit_times, qubit_phases)

    def duration(self, qubit_config):
        return RX.duration(self, qubit_config)

    def to_sequence(self, sequence):
        q = self.target_qubits[0]
        theta = self.parameters
        phi = 0
        lam = 0
        sequence.add_u3(theta, phi, lam)


class RZ(AbstractHardwareGate, gates.RZ):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases): # pragma: no cover
        raise_error(NotImplementedError)

    def duration(self, qubit_config): # pragma: no cover
        raise_error(NotImplementedError)

    def to_sequence(self, sequence):
        # apply virtually by changing ``phase`` instead of using pulses
        sequence.phase += self.parameters
        #q = self.target_qubits[0]
        #theta = 0
        #phi = self.parameters / 2
        #lam = self.parameters / 2
        #return sequence.add_u3(theta, phi, lam)


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

    def to_sequence(self, sequence):
        raise_error(NotImplementedError)


class U2(AbstractHardwareGate, gates.U2):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases): # pragma: no cover
        raise_error(NotImplementedError)

    def duration(self, qubit_config): # pragma: no cover
        raise_error(NotImplementedError)

    def to_sequence(self, sequence):
        args = (math.pi / 2,) + self.parameters
        sequence.add_u3(*args)


class U3(AbstractHardwareGate, gates.U3):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases): # pragma: no cover
        raise_error(NotImplementedError)

    def duration(self, qubit_config): # pragma: no cover
        raise_error(NotImplementedError)

    def to_sequence(self, sequence):
        sequence.add_u3(*self.parameters)


class X(AbstractHardwareGate, gates.X):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases): # pragma: no cover
        raise_error(NotImplementedError)

    def duration(self, qubit_config): # pragma: no cover
        raise_error(NotImplementedError)

    def to_sequence(self, sequence):
        sequence.add_u3(math.pi, 0, math.pi)


class Y(AbstractHardwareGate, gates.Y):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases): # pragma: no cover
        raise_error(NotImplementedError)

    def duration(self, qubit_config): # pragma: no cover
        raise_error(NotImplementedError)

    def to_sequence(self, sequence):
        sequence.add_u3(math.pi, 0, 0)


class Z(AbstractHardwareGate, gates.Z):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases): # pragma: no cover
        raise_error(NotImplementedError)

    def duration(self, qubit_config): # pragma: no cover
        raise_error(NotImplementedError)

    def to_sequence(self, sequence):
        sequence.add_u3(0, math.pi, 0)
