from collections.abc import Sequence
from typing import Optional

from qibolab._core.identifier import ChannelId
from qibolab._core.instruments.qblox.sequence.asm import label
from qibolab._core.pulses.pulse import PulseId
from qibolab._core.serialize import Model
from qibolab._core.sweeper import ParallelSweepers, iteration_length

from ..q1asm.ast_ import (
    Add,
    Block,
    BlockIter,
    BlockList,
    Instruction,
    Jlt,
    Line,
    Loop,
    Move,
    Nop,
    Reference,
    Register,
    ResetPh,
    Sub,
    Wait,
)
from .asm import Registers
from .sweepers import IndexedParams, Param, update_instructions

__all__ = []


class LoopSpec(Model):
    """Loop descriptor.

    Created by the :func:`loops` function.
    """

    reg: Register
    """Register used for the loop counter."""
    length: int
    """Iteration length."""
    id: Optional[int]
    """Iteration index.

    `None` marks the loop as the shots loop.
    """

    @property
    def shots(self) -> bool:
        return self.id is None

    @property
    def description(self) -> str:
        """Sweeper textual description."""
        return f"loop {self.id + 1}" if self.id is not None else "shots"


def loops(
    sweepers: list[ParallelSweepers], nshots: int, inner_shots: bool
) -> Sequence[LoopSpec]:
    """Initialize registers for loop counters.

    The counters implement the ``length`` of the iteration, which, for a general
    sweeper, is fully characterized by a ``(start, step, length)`` tuple.

    Those related to :attr:`Registers.bin` and :attr:`Registers.bin_reset` are actually
    not loop counter on their own, but they are required to properly store the
    acquisitions in different bins.
    """
    shots = LoopSpec(reg=Registers.shots.value, length=nshots, id=None)
    sweep = [
        LoopSpec(
            reg=Register(number=i + Registers.first_available()),
            length=iteration_length(parsweep),
            id=i,
        )
        # the first sweeper should be the outermost, thus reverse them during the
        # enumeration
        for i, parsweep in enumerate(sweepers[::-1])
    ]
    return [shots] + sweep if inner_shots else sweep + [shots]


START: str = "start"
"""Label first experiment's instruction."""
SHOTS: str = "shots"
"""Label instruction after bin counter reset instruction.

The instruction is used for shots, and skipped when all shots are
finished.

Cf. :const:`_SHOTS_BIN_RESET`.
"""


def _experiment_end(relaxation_time: int) -> list[Line]:
    """Wrap up experiment.

    - relax
    - reset phase (?)
    - increment bin where the result is saved

    The bin increment is possibly reset later on, in order to save on the same bin, and
    thus summing various shots (eventually averaging). Cf. :const:`_SHOTS_BIN_RESET`.
    """
    return [
        Line(instruction=Wait(duration=relaxation_time), comment="relaxation"),
        Line(instruction=ResetPh(), comment="phase reset"),
        Line(
            instruction=Add(
                a=Registers.bin.value, b=1, destination=Registers.bin.value
            ),
            comment="bin increment",
        ),
    ]


_SHOTS_BIN_RESET: list[Line] = [
    Line(
        instruction=Jlt(a=Registers.shots.value, b=1, address=Reference(label=SHOTS)),
        comment="skip bin reset - advance both counters",
    ),
    Line(
        instruction=Move(
            source=Registers.bin_reset.value,
            destination=Registers.bin.value,
        ),
        comment="shots average: reset bin counter",
    ),
]
"""Reset bin counter to save shots to the same one.

The full algorithm is detailed at:
https://github.com/qiboteam/qibolab/discussions/1119
"""


def _sweep_update(p: Param, channel: set[ChannelId], pulses: set[PulseId]) -> Block:
    """Sweeper update for a single parameter.

    - increment the parameter register
    - set new parameter value (if channel-wise)
        - an additional `nop` instruction is plugged to wait one further clock cycle in
          between the register increment and the parameter update, to ensure the value
          is correctly propagated
          https://docs.qblox.com/en/main/products/architecture/sequencers/sequencer.html#registers
    """
    return (
        *(
            (
                Line(
                    instruction=(
                        Add(a=p.reg, b=p.step, destination=p.reg)
                        if p.step >= 0
                        else Sub(a=p.reg, b=-p.step, destination=p.reg)
                    ),
                    comment=f"shift {p.description}",
                ),
            )
            if p.channel in channel or p.pulse in pulses
            else ()
        ),
        *(
            (
                # wait one more clock cycle
                [Nop()]
                # then update the value
                + update_instructions(p.role, p.reg)
            )
            if p.channel in channel
            else ()
        ),
    )


def _sweep_updates(
    lp: LoopSpec, params: IndexedParams, channel: set[ChannelId], pulses: set[PulseId]
) -> BlockIter:
    """Parallel sweeper updates.

    Collects all synced updates, those happening in a single loop.
    """
    return (
        (
            line
            for block in (
                _sweep_update(p, channel, pulses) for p in (params[lp.id].all)
            )
            for line in block
        )
        if lp.id is not None
        else ()
    )


def _sweep_reset(
    params: list[Param], channel: set[ChannelId], pulses: set[PulseId]
) -> BlockList:
    """Reset sweeper register value.

    Once the loop is completed, the parameter value, which is hold in the respective
    register, needs to be reset to its original value. To be ready when a new loop will
    possibly start (which is always the case, but for the outermost loop).

    .. note::

        Channel parameters are also updated immediately, in order for the change to take
        effect.
        Pulse parameters will anyhow act around the suitable pulse, so the update is
        always performed when needed.
    """
    return [
        Line(
            instruction=Move(source=p.start, destination=p.reg),
            comment=f"init {p.description}",
        )
        for p in params
        if p.channel in channel or p.pulse in pulses
    ] + [
        inst
        for p in params
        if p.channel in channel
        for inst in update_instructions(p.role, p.reg)
    ]


def _sweep_iteration(
    lp: LoopSpec,
    params: IndexedParams,
    shots: bool,
    channel: set[ChannelId],
    pulses: set[PulseId],
) -> BlockList:
    """Sweep loop.

    The operations performed by the looping machinery are divided in internal to loop,
    and those happening right after completing all loops.

    The internal ones consist of:

    - update the parameters for all the sweepers set at the selected loop level
    - loop, i.e. decrement iteration counter and jump back to experiment start

    While the closing operations include:

    - reset iteration counter, after a whole cycle is completed
    - reset the parameters' values, to possibly get ready for an entire new iteration,
      triggered by an external loop
    """
    loops_ = [
        *_sweep_updates(lp, params, channel, pulses),
        Line(
            instruction=Loop(a=lp.reg, address=Reference(label=START)),
            comment=f"loop over {lp.description}",
            label=SHOTS if shots else None,
        ),
        Move(source=lp.length, destination=lp.reg),
    ]
    # no parameter reset for the shots loop, since, by definition, that's just an unaltered
    # repetition of the experiment
    reset = [] if lp.id is None else _sweep_reset(params[lp.id].all, channel, pulses)
    return loops_ + reset


def _loop_machinery(
    loops: Sequence[LoopSpec],
    params: IndexedParams,
    singleshot: bool,
    channel: set[ChannelId],
    pulses: set[PulseId],
) -> BlockList:
    """Looping block.

    It creates an instruction block to be entirely placed after the
    experiment (including the relaxation part) to repeat it as
    specified, taking into account both shots and nested sweeper loops.
    """

    def shots(marker: LoopSpec) -> bool:
        return marker.shots and not singleshot

    return [
        inst
        for lp in loops
        for inst in (
            (_SHOTS_BIN_RESET if shots(lp) else [])
            + _sweep_iteration(lp, params, shots(lp), channel, pulses)
        )
    ][:-1]


def loop(
    experiment: list[Instruction],
    loops: Sequence[LoopSpec],
    params: IndexedParams,
    relaxation_time: int,
    singleshot: bool,
    channel: set[ChannelId],
    pulses: set[PulseId],
) -> Block:
    end = _experiment_end(relaxation_time)
    machinery = _loop_machinery(loops, params, singleshot, channel, pulses)
    main = experiment + end + machinery

    return [label(main[0], START)] + main[1:]
