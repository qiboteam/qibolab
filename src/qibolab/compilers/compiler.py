from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from qibo import gates
from qibo.config import raise_error

from qibolab.compilers.default import (
    cz_rule,
    identity_rule,
    measurement_rule,
    rz_rule,
    u3_rule,
    z_rule,
)
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

    rules: dict
    """Map from gates to compilation rules."""
    measurement_map: dict = field(default_factory=dict)
    """Map from each measurement gate to the sequence of readout pulses implementing it."""

    @classmethod
    def default(cls):
        return cls(
            {
                gates.I: identity_rule,
                gates.Z: z_rule,
                gates.RZ: rz_rule,
                gates.U3: u3_rule,
                gates.CZ: cz_rule,
                gates.M: measurement_rule,
            }
        )

    def __setitem__(self, key, rule):
        """Sets a new rule to the compiler.

        If a rule already exists for the gate, it will be overwritten.
        """
        self.rules[key] = rule

    def __getitem__(self, item):
        """Get an existing rule for a given gate."""
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

                    # register readout sequences to ``measurement_map`` so that we can
                    # properly map acquisition results to measurement gates
                    if isinstance(gate, gates.M):
                        self.measurement_map[gate] = gate_sequence

                    already_processed.add(gate)

        return sequence

    def assign_measurements(self, backend, readout):
        """Assign measurement outcomes to ``CircuitResult`` object.

        Args:
            backend (:class:`qibo.backends.abstract.AbstractBackend`): Backend object to
                be assigned in the result object.
            readout (:class:`qibolab.results.ExecutionResults`): Acquisition results
                containing shot values.
        """
        for gate, sequence in self.measurement_map.items():
            samples = []
            for pulse in sequence.pulses:
                shots = readout[pulse.serial].shots
                if shots is not None:
                    samples.append(shots)
            # TODO: Check in qibo why backend is needed to be assigned here
            gate.result.backend = backend
            gate.result.register_samples(np.array(samples).T)
