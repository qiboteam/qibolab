from collections import defaultdict
from dataclasses import dataclass, field

from qibo.config import raise_error

from qibolab.pulses import PulseSequence, ReadoutPulse


@dataclass
class Compiler:
    """Compiler that transforms a :class:`qibo.models.Circuit` to a :class:`qibolab.pulses.PulseSequence`.

    The transformation is done using a dictionary of rules which map each Qibo gate to a
    pulse sequence and some virtual Z-phases.

    A rule is a function that takes two argumens:
        gate (:class:`qibo.gates.abstract.Gate`): Gate object to be compiled.
        platform (:class:`qibolab.platforms.abstract.AbstractPlatform`): Platform object to read
            native gate pulses from.
    and returns
        sequence (:class:`qibolab.pulses.PulseSequence`): Sequence of pulses that implement
            the given gate.
        virtual_z_phases (dict): Dictionary mapping qubits to virtual Z-phases induced by the gate.

    See ``qibolab.compilers.default`` for an example of a compiler implementation.
    """

    rules: dict = field(default_factory=dict)

    def __setitem__(self, key, rule):
        """Sets a new rule to the compiler.

        If a rule already exists for the gate, it will be overwritten.
        """
        self.rules[key] = rule

    def __getitem__(self, item):
        if item not in self.rules:
            raise_error(KeyError, f"Compiler rule not available for {item}.")
        return self.rules[item]

    def register(self, gate_cls):
        """Decorator for registering a function as a rule in the compiler.

        Using this decorator is optional. Alternatively the user can set the rules directly
        via ``__setitem__`.

        Args:
            gate_cls: Qibo gate object that the rule will be assigned to.
        """

        def inner(func):
            self[gate_cls] = func
            return func

        return inner

    def remove(self, item):
        """Remove rule for the given gate."""
        if item not in self.rules:
            raise_error(KeyError, f"Cannot remove {item} from compiler because it does not exist.")
        self.rules.pop(item)

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
