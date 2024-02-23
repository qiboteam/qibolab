from collections import defaultdict
from dataclasses import dataclass, field

from qibo import gates
from qibo.config import raise_error

from qibolab.compilers.default import (
    cnot_rule,
    cz_rule,
    gpi2_rule,
    gpi_rule,
    identity_rule,
    measurement_rule,
    rz_rule,
    u3_rule,
    z_rule,
)
from qibolab.pulses import Delay, PulseSequence, PulseType


@dataclass
class Compiler:
    """Compiler that transforms a :class:`qibo.models.Circuit` to a
    :class:`qibolab.pulses.PulseSequence`.

    The transformation is done using a dictionary of rules which map each Qibo gate to a
    pulse sequence and some virtual Z-phases.

    A rule is a function that takes two argumens:
        - gate (:class:`qibo.gates.abstract.Gate`): Gate object to be compiled.
        - platform (:class:`qibolab.platforms.abstract.AbstractPlatform`): Platform object to read
            native gate pulses from.

    and returns:
        - sequence (:class:`qibolab.pulses.PulseSequence`): Sequence of pulses that implement
            the given gate.
        - virtual_z_phases (dict): Dictionary mapping qubits to virtual Z-phases induced by the gate.

    See :class:`qibolab.compilers.default` for an example of a compiler implementation.
    """

    rules: dict = field(default_factory=dict)
    """Map from gates to compilation rules."""

    @classmethod
    def default(cls):
        return cls(
            {
                gates.I: identity_rule,
                gates.Z: z_rule,
                gates.RZ: rz_rule,
                gates.U3: u3_rule,
                gates.CZ: cz_rule,
                gates.CNOT: cnot_rule,
                gates.GPI2: gpi2_rule,
                gates.GPI: gpi_rule,
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
        try:
            return self.rules[item]
        except KeyError:
            raise_error(KeyError, f"Compiler rule not available for {item}.")

    def __delitem__(self, item):
        """Remove rule for the given gate."""
        try:
            del self.rules[item]
        except KeyError:
            raise_error(
                KeyError,
                f"Cannot remove {item} from compiler because it does not exist.",
            )

    def register(self, gate_cls):
        """Decorator for registering a function as a rule in the compiler.

        Using this decorator is optional. Alternatively the user can set the rules directly
        via ``__setitem__``.

        Args:
            gate_cls: Qibo gate object that the rule will be assigned to.
        """

        def inner(func):
            self[gate_cls] = func
            return func

        return inner

    def compile(self, circuit, platform):
        """Transforms a circuit to pulse sequence.

        Args:
            circuit (qibo.models.Circuit): Qibo circuit that respects the platform's
                                           connectivity and native gates.
            platform (qibolab.platforms.abstract.AbstractPlatform): Platform used
                to load the native pulse representations.

        Returns:
            sequence (qibolab.pulses.PulseSequence): Pulse sequence that implements the circuit.
            measurement_map (dict): Map from each measurement gate to the sequence of  readout pulse implementing it.
        """
        sequence = PulseSequence()
        # FIXME: This will not work with qubits that have string names
        # TODO: Implement a mapping between circuit qubit ids and platform ``Qubit``s
        virtual_z_phases = defaultdict(int)

        measurement_map = {}
        qubit_clock = defaultdict(int)
        channel_clock = defaultdict(int)
        # process circuit gates
        for moment in circuit.queue.moments:
            for gate in set(filter(lambda x: x is not None, moment)):
                if isinstance(gate, gates.Align):
                    for qubit in gate.qubits:
                        # TODO: do something
                        pass
                    continue

                rule = self[gate.__class__]
                # get local sequence and phases for the current gate
                gate_sequence, gate_phases = rule(gate, platform)
                for pulse in gate_sequence:
                    if pulse.type is not PulseType.READOUT:
                        pulse.relative_phase += virtual_z_phases[pulse.qubit]

                    if qubit_clock[pulse.qubit] > channel_clock[pulse.qubit]:
                        delay = qubit_clock[pulse.qubit] - channel_clock[pulse.channel]
                        sequence.append(Delay(delay, pulse.channel))
                        channel_clock[pulse.channel] += delay

                    sequence.append(pulse)
                    # update clocks
                    qubit_clock[pulse.qubit] += pulse.duration
                    channel_clock[pulse.channel] += pulse.duration

                # update virtual Z phases
                for qubit, phase in gate_phases.items():
                    virtual_z_phases[qubit] += phase

                # register readout sequences to ``measurement_map`` so that we can
                # properly map acquisition results to measurement gates
                if isinstance(gate, gates.M):
                    measurement_map[gate] = gate_sequence

        return sequence, measurement_map
