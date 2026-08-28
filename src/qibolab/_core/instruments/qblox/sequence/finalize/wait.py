from qibolab._core.serialize import Model

from ...q1asm.ast_ import Block, Line, Loop, Move, Reference, Wait
from ..asm import Registers
from .components import BlockRule, LineRule

__all__ = [
    "LongWaits",
    "long_wait",
    "merge_wait",
]

MAX_WAIT = 2**16 - 1


class LongWaits(Model):
    n: int = 0
    """Number of long waits expanded."""


def _match_long_wait(line: Line, state: LongWaits) -> tuple[bool, LongWaits]:
    instr = line.instruction
    match = (
        isinstance(instr, Wait)
        and isinstance(instr.duration, int)
        and instr.duration > MAX_WAIT
    )
    return match, state


def _map_long_wait(line: Line, state: LongWaits) -> tuple[Block, LongWaits]:
    instr = line.instruction
    assert isinstance(instr, Wait)
    duration = instr.duration
    assert isinstance(duration, int)

    iterations = duration // MAX_WAIT
    remainder = duration % MAX_WAIT
    register = Registers.wait.value
    label = f"wait{state.n}"

    block = ([Wait(duration=remainder)] if remainder > 0 else []) + [
        Move(source=iterations, destination=register),
        Line(instruction=Wait(duration=MAX_WAIT), label=label),
        Line.instr(Loop(a=register, address=Reference(label=label))),
    ]

    return block, LongWaits(n=state.n + 1)


long_wait = LineRule[LongWaits](match=_match_long_wait, map=_map_long_wait)
"""Split a long immediate wait.

It accounts for the wait instruction limit, defined by :const:`MAX_WAIT`.

Each loop is labelled with its own unique tag, which is assigned sequentially by keeping
the count of expanded loops.
"""


def _start_merge_wait(line: Line, state: None) -> tuple[bool, None]:
    instr = line.instruction
    intwait = isinstance(instr, Wait) and isinstance(instr.duration, int)
    return intwait, None


def _end_merge_wait(line: Line, state: None) -> tuple[bool, None]:
    instr = line.instruction
    intwait = isinstance(instr, Wait) and isinstance(instr.duration, int)
    return not intwait or line.label is not None, None


def _merge_wait(lines: list[Line], state: None) -> tuple[Block, None]:
    """Merge subsequent static (immediate) waits."""
    duration: int = 0
    comment: list[str] = []
    label: str | None = lines[0].label

    for line in lines:
        instr = line.instruction
        assert isinstance(instr, Wait) and isinstance(instr.duration, int)

        duration += instr.duration
        if line.comment is not None:
            comment.append(line.comment)
        if line.label is not None:
            label = line.label

    merged = (
        [
            Line(
                label=label,
                instruction=Wait(duration=duration),
                comment="\n".join(comment),
            )
        ]
        if duration > 0
        else []
    )
    return merged, None


merge_wait = BlockRule[None](
    initial=_start_merge_wait, final=_end_merge_wait, map=_merge_wait
)
