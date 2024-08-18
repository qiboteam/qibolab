from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

from qibo import Circuit, gates

from qibolab.compilers.default import (
    align_rule,
    cnot_rule,
    cz_rule,
    gpi2_rule,
    gpi_rule,
    identity_rule,
    measurement_rule,
    rz_rule,
    z_rule,
)
from qibolab.platform import Platform
from qibolab.pulses import Delay
from qibolab.qubits import QubitId
from qibolab.sequence import PulseSequence

Rule = Callable[..., PulseSequence]
"""Compiler rule."""


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

    rules: dict[type[gates.Gate], Rule] = field(default_factory=dict)
    """Map from gates to compilation rules."""

    @classmethod
    def default(cls):
        return cls(
            {
                gates.I: identity_rule,
                gates.Z: z_rule,
                gates.RZ: rz_rule,
                gates.CZ: cz_rule,
                gates.CNOT: cnot_rule,
                gates.GPI2: gpi2_rule,
                gates.GPI: gpi_rule,
                gates.M: measurement_rule,
                gates.Align: align_rule,
            }
        )

    def register(self, gate_cls: type[gates.Gate]) -> Callable[[Rule], Rule]:
        """Decorator for registering a function as a rule in the compiler.

        Using this decorator is optional. Alternatively the user can set the rules directly
        via ``__setitem__``.

        Args:
            gate_cls: Qibo gate object that the rule will be assigned to.
        """

        def inner(func: Rule) -> Rule:
            self.rules[gate_cls] = func
            return func

        return inner

    def get_sequence(self, gate: gates.Gate, platform: Platform) -> PulseSequence:
        """Get pulse sequence implementing the given gate using the registered
        rules.

        Args:
            gate (:class:`qibo.gates.Gate`): Qibo gate to convert to pulses.
            platform (:class:`qibolab.platform.Platform`): Qibolab platform to read the native gates from.
        """
        # get local sequence for the current gate
        rule = self.rules[type(gate)]

        natives = platform.natives

        if isinstance(gate, (gates.M)):
            qubits = [
                natives.single_qubit[platform.get_qubit(q).name] for q in gate.qubits
            ]
            return rule(gate, qubits)

        if isinstance(gate, (gates.Align)):
            qubits = [platform.get_qubit(q) for q in gate.qubits]
            return rule(gate, qubits)

        if isinstance(gate, (gates.Z, gates.RZ)):
            qubit = platform.get_qubit(gate.target_qubits[0])
            return rule(gate, qubit)

        if len(gate.qubits) == 1:
            qubit = platform.get_qubit(gate.target_qubits[0])
            return rule(gate, natives.single_qubit[qubit.name])

        if len(gate.qubits) == 2:
            pair = tuple(platform.get_qubit(q).name for q in gate.qubits)
            assert len(pair) == 2
            return rule(gate, natives.two_qubit[pair])

        raise NotImplementedError(f"{type(gate)} is not a native gate.")

    # FIXME: pulse.qubit and pulse.channel do not exist anymore
    def compile(
        self, circuit: Circuit, platform: Platform
    ) -> tuple[PulseSequence, dict[gates.M, PulseSequence]]:
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
        ch_to_qb = platform.channels_map

        sequence = PulseSequence()
        # FIXME: This will not work with qubits that have string names
        # TODO: Implement a mapping between circuit qubit ids and platform ``Qubit``s

        measurement_map = {}
        channel_clock = defaultdict(float)

        def qubit_clock(el: QubitId):
            elements = platform.qubits if el in platform.qubits else platform.couplers
            return max(channel_clock[ch.name] for ch in elements[el].channels)

        # process circuit gates
        for moment in circuit.queue.moments:
            for gate in {x for x in moment if x is not None}:
                delay_sequence = PulseSequence()
                gate_sequence = self.get_sequence(gate, platform)
                increment = defaultdict(float)
                active_qubits = {ch_to_qb[ch] for ch in gate_sequence.channels}
                start = max((qubit_clock(el) for el in active_qubits), default=0.0)
                for ch in gate_sequence.channels:
                    delay = start - channel_clock[ch]
                    if delay > 0:
                        delay_sequence.append((ch, Delay(duration=delay)))
                        channel_clock[ch] += delay
                    increment[ch] = gate_sequence.channel_duration(ch)
                for q in gate.qubits:
                    qubit = platform.get_qubit(q)
                    if qubit not in active_qubits:
                        increment[qubit.drive] = (
                            start + gate_sequence.duration - channel_clock[qubit.drive]
                        )

                # add the increment only after computing them, since multiple channels
                # are related to each other because belonging to the same qubit
                for ch, inc in increment.items():
                    channel_clock[ch] += inc
                sequence.concatenate(delay_sequence)
                sequence.concatenate(gate_sequence)

                # register readout sequences to ``measurement_map`` so that we can
                # properly map acquisition results to measurement gates
                if isinstance(gate, gates.M):
                    measurement_map[gate] = gate_sequence

        return sequence.trim(), measurement_map
