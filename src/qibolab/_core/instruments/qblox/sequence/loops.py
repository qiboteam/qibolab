from collections.abc import Sequence
from enum import Enum
from typing import Optional, Union, cast

from qibolab._core.identifier import ChannelId
from qibolab._core.serialize import Model
from qibolab._core.sweeper import ParallelSweepers, iteration_length

from ..q1asm.ast_ import (
    Add,
    Block,
    Instruction,
    Jge,
    Line,
    Loop,
    Move,
    Reference,
    Register,
    ResetPh,
    Wait,
)
from .sweepers import IndexedParams, update_instructions

__all__ = []


class Registers(Enum):
    bin = Register(number=0)
    bin_reset = Register(number=1)
    shots = Register(number=2)


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
        return f"sweeper {self.id + 1}" if self.id is not None else "shots"


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
    first_sweeper = max(r.value.number for r in Registers) + 1
    sweep = [
        LoopSpec(
            reg=Register(number=i + first_sweeper),
            length=iteration_length(parsweep),
            id=i,
        )
        for i, parsweep in enumerate(sweepers)
    ]
    return [shots] + sweep if inner_shots else sweep + [shots]


START = "start"
SHOTS = "shots"


def _experiment_end(relaxation_time: int) -> list[Line]:
    """Wrap up experiment.

    - relax
    - reset phase (?)
    - increment bin where the result is saved

    The bin increment is possibly reset later on, in order to save on the same bin, and
    thus summing various shots (eventually averaging). Cf. :func:`_shots_bin_reset`.
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


def _shots_bin_reset(lp, singleshot: bool) -> list[Line]:
    """Reset bin counter to save shots to the same one.

    The full algorithm is detailed at:
    https://github.com/qiboteam/qibolab/discussions/1119
    """
    return [
        Line(
            instruction=Jge(
                a=Registers.shots.value, b=1, address=Reference(label=SHOTS)
            ),
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


def _sweep_iteration(
    lp: LoopSpec, params, shots: bool, channel: ChannelId
) -> list[Union[Line, Instruction]]:
    """Sweep loop.

    - increment the parameter register
    - update parameter (if channel-wise)
    - loop, i.e. decrement iteration counter and jump
    - reset iteration counter, after a whole cycle is completed
    """
    return [
        *(
            (
                line
                for block in (
                    (
                        Line(
                            instruction=Add(a=p.reg, b=p.step, destination=p.reg),
                            comment=f"increment {p.description}",
                        ),
                        *(
                            update_instructions(p.kind, p.reg)
                            if p.description is not None and p.channel == channel
                            else ()
                        ),
                    )
                    for p in (params[lp.id][0] + params[lp.id][1])
                )
                for line in block
            )
            if lp.id is not None
            else ()
        ),
        Line(
            instruction=Loop(a=lp.reg, address=Reference(label=START)),
            comment=f"loop over {lp.description}",
            label=SHOTS if shots else None,
        ),
        Move(source=lp.length, destination=lp.reg),
    ]


def _loop_machinery(
    loops: Sequence[LoopSpec],
    params: IndexedParams,
    singleshot: bool,
    channel: ChannelId,
) -> Block:
    """Looping block.

    It creates an instruction block to be entirely placed after the
    experiment (including the relaxation part) to repeat it as
    specified, taking into account both shots and nested sweeper loops.
    """

    def shots(marker: LoopSpec) -> bool:
        return marker.shots and not singleshot

    return [
        i_
        for lp in loops
        for i_ in (
            (_shots_bin_reset(lp, singleshot) if shots(lp) else [])
            + _sweep_iteration(lp, params, shots(lp), channel)
        )
    ][:-1]


def loop(
    loops: Sequence[LoopSpec],
    params: IndexedParams,
    experiment: list[Instruction],
    relaxation_time: int,
    singleshot: bool,
    channel: ChannelId,
) -> Block:
    end = cast(list, _experiment_end(relaxation_time))
    machinery = cast(list, _loop_machinery(loops, params, singleshot, channel))
    main = experiment + end + machinery

    return [
        (
            Line(instruction=main[0], label=START)
            if isinstance(main[0], Instruction)
            else Line(
                instruction=main[0].instruction, comment=main[0].comment, label=START
            )
        )
    ] + main[1:]
