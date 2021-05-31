import sys
import math
import copy
from abc import ABC, abstractmethod
from qibo import gates
from qibo.config import raise_error


class AbstractHardwareGate(ABC):
    module = sys.modules[__name__]

    @abstractmethod
    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        raise_error(NotImplementedError)

    @abstractmethod
    def duration(self, qubit_config):
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


class I(AbstractHardwareGate, gates.I):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        return []

    def duration(self, qubit_config):
        return 0


class Align(AbstractHardwareGate, gates.Align):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        m = max(qubit_times[q] for q in self.target_qubits)
        for q in self.target_qubits:
            qubit_times[q] = m
        return []

    def duration(self, qubit_config):
        return 0


class M(AbstractHardwareGate, gates.M):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        pulses = []
        for q in self.target_qubits:
            pulses += qubit_config[q]["gates"][self.name]
        return pulses

    def duration(self, qubit_config):
        q = self.target_qubits[0]
        pulses = qubit_config[q]["gates"][self.name]
        m = 0
        for p in pulses:
            m = max(p.duration, m)
        return m


class RX(AbstractHardwareGate, gates.RX):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        if self.parameters == 0:
            return []

        q = self.target_qubits[0]
        time_mod = abs(self.parameters / math.pi)
        phase_mod = 0 if self.parameters > 0 else -180
        phase_mod += qubit_phases[q]
        m = 0

        pulses = copy.deepcopy(qubit_config[q]["gates"][self.name])
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
        pulses = qubit_config[q]["gates"][self.name]
        m = 0

        for p in pulses:
            m = max(p.duration * time_mod, m)
        return m


class RY(AbstractHardwareGate, gates.RY):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        return RX.pulse_sequence(self, qubit_config, qubit_times, qubit_phases)

    def duration(self, qubit_config):
        return RX.duration(self, qubit_config)


class CNOT(AbstractHardwareGate, gates.CNOT):

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        q = self.target_qubits[0]
        control = self.control_qubits[0]
        start = max(qubit_times[q], qubit_times[control])
        pulses = copy.deepcopy(qubit_config[q]["gates"][self.name + "_{}".format(control)])

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
