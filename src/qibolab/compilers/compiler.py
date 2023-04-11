from collections import defaultdict
from dataclasses import dataclass, field

from qibo.config import raise_error

from qibolab.pulses import PulseSequence, ReadoutPulse


@dataclass
class Compiler:
    rules: dict = field(default_factory=dict)

    def register(self, gate_cls):
        def inner(func):
            self.rules[gate_cls] = func
            return func

        return inner

    def remove(self, item):
        if item not in self.rules:
            raise_error(KeyError, f"Cannot remove {item} from compiler because it does not exist.")
        self.rules.pop(item)

    def __setitem__(self, key, rule):
        self.rules[key] = rule

    def __getitem__(self, item):
        if item not in self.rules:
            raise_error(KeyError, f"Compiler rule not available for {item}.")
        return self.rules[item]

    def __call__(self, circuit, platform):
        """Transforms a circuit to pulse sequence.

        Args:
            circuit (qibo.models.Circuit): Qibo circuit that respects the platform's
                connectivity and native gates.
            platform (qibolab.platforms.abstract.AbstractPlatform): Platform used
                to load the native pulse representations.

        Returns:
            sequence (qibolab.pulses.PulseSequence): Pulse sequence that implements the circuit.
        """
        sequence = PulseSequence()
        # FIXME: This will not work with qubits that have string names
        # TODO: Implement a mapping between circuit qubit ids and platform ``Qubit``s
        virtual_z_phases = defaultdict(int)

        # keep track of gates that were already added to avoid adding them twice
        already_processed = set()
        # process circuit gates
        for moment in circuit.queue.moments:
            moment_start = sequence.finish
            for gate in moment:
                if gate is not None and gate not in already_processed:
                    rule = self[gate.__class__]
                    # get local sequence and phases for the current gate
                    gate_sequence, gate_phases = rule(gate, platform)

                    # update global pulse sequence
                    # determine the right start time based on the availability of the qubits involved
                    all_qubits = {*gate_sequence.qubits, *gate.qubits}
                    start = max(sequence.get_qubit_pulses(*all_qubits).finish, moment_start)
                    # shift start time and phase according to the global sequence
                    for pulse in gate_sequence:
                        pulse.start += start
                        if not isinstance(pulse, ReadoutPulse):
                            pulse.relative_phase += virtual_z_phases[pulse.qubit]
                        sequence.add(pulse)

                    # update virtual Z phases
                    for qubit, phase in gate_phases.items():
                        virtual_z_phases[qubit] += phase

                    already_processed.add(gate)

        return sequence
