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
from qibolab.identifier import ChannelId, QubitId
from qibolab.platform import Platform
from qibolab.pulses import Delay
from qibolab.sequence import PulseSequence

Rule = Callable[..., PulseSequence]
"""Compiler rule."""


@dataclass
class Compiler:
    """Compile native circuits into pulse sequences.

    It transforms a :class:`qibo.models.Circuit` to a :class:`qibolab.pulses.PulseSequence`.

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
        """Register a function as a rule in the compiler.

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
        """Get pulse sequence implementing the given gate.

        The sequence is obtained using the registered rules.

        Args:
            gate (:class:`qibo.gates.Gate`): Qibo gate to convert to pulses.
            platform (:class:`qibolab.platform.Platform`): Qibolab platform to read the native gates from.
        """
        # get local sequence for the current gate
        rule = self.rules[type(gate)]

        natives = platform.natives

        if isinstance(gate, (gates.M)):
            qubits = [natives.single_qubit[platform.qubit(q)[0]] for q in gate.qubits]
            return rule(gate, qubits)

        if isinstance(gate, (gates.Align)):
            qubits = [platform.qubit(q)[1] for q in gate.qubits]
            return rule(gate, qubits)

        if isinstance(gate, (gates.Z, gates.RZ)):
            qubit = platform.qubit(gate.target_qubits[0])[1]
            return rule(gate, qubit)

        if len(gate.qubits) == 1:
            qubit = platform.qubit(gate.target_qubits[0])[0]
            return rule(gate, natives.single_qubit[qubit])

        if len(gate.qubits) == 2:
            pair = tuple(platform.qubit(q)[0] for q in gate.qubits)
            assert len(pair) == 2
            return rule(gate, natives.two_qubit[pair])

        raise NotImplementedError(f"{type(gate)} is not a native gate.")

    def _compile_gate(
        self,
        gate: gates.Gate,
        platform: Platform,
        channel_clock: defaultdict[ChannelId, float],
    ) -> PulseSequence:
        def qubit_clock(el: QubitId):
            return max(channel_clock[ch] for ch in platform.qubits[el].channels)

        def coupler_clock(el: QubitId):
            return max(channel_clock[ch] for ch in platform.couplers[el].channels)

        gate_seq = self.get_sequence(gate, platform)
        # qubits receiving pulses
        qubits = {
            q
            for q in [platform.qubit_channels.get(ch) for ch in gate_seq.channels]
            if q is not None
        }
        # couplers receiving pulses
        couplers = {
            c
            for c in [platform.coupler_channels.get(ch) for ch in gate_seq.channels]
            if c is not None
        }

        # add delays to pad all involved channels to begin at the same time
        start = max(
            [qubit_clock(q) for q in qubits] + [coupler_clock(c) for c in couplers],
            default=0.0,
        )
        initial = PulseSequence()
        for ch in gate_seq.channels:
            delay = start - channel_clock[ch]
            if delay > 0:
                initial.append((ch, Delay(duration=delay)))
            channel_clock[ch] = start + gate_seq.channel_duration(ch)

        # pad all qubits to have at least one channel busy for the duration of the gate
        # (drive arbitrarily chosen, as always present)
        end = start + gate_seq.duration
        final = PulseSequence()
        for q in gate.qubits:
            qubit = platform.qubit(q)[1]
            # all actual qubits have a non-null drive channel, and couplers are not
            # explicitedly listed in gates
            assert qubit.drive is not None
            delay = end - channel_clock[qubit.drive]
            if delay > 0:
                final.append((qubit.drive, Delay(duration=delay)))
                channel_clock[qubit.drive] += delay
        # couplers do not require individual padding, because they do are only
        # involved in gates where both of the other qubits are involved

        return initial + gate_seq + final

    def compile(
        self, circuit: Circuit, platform: Platform
    ) -> tuple[PulseSequence, dict[gates.M, PulseSequence]]:
        """Transform a circuit to pulse sequence.

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

        measurement_map = {}
        channel_clock = defaultdict(float)

        # process circuit gates
        for moment in circuit.queue.moments:
            for gate in {x for x in moment if x is not None}:
                sequence += self._compile_gate(gate, platform, channel_clock)

                # register readout sequences to ``measurement_map`` so that we can
                # properly map acquisition results to measurement gates
                if isinstance(gate, gates.M):
                    measurement_map[gate] = self.get_sequence(gate, platform)

        return sequence.trim(), measurement_map
