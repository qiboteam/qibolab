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
    instr = line.instr
    match = (
        isinstance(instr, Wait)
        and isinstance(instr.duration, int)
        and instr.duration > MAX_WAIT
    )
    return match, state


def _map_long_wait(line: Line, state: LongWaits) -> tuple[Block, LongWaits]:
    instr = line.instr
    assert isinstance(instr, Wait)
    duration = instr.duration
    assert isinstance(duration, int)

    iterations = duration // MAX_WAIT
    remainder = duration % MAX_WAIT
    register = Registers.wait.value
    label = f"wait{state.n}"

    block = [Wait(duration=remainder)] + [
        Move(source=iterations, destination=register),
        Line(instruction=Wait(duration=MAX_WAIT), label=label),
        Line.instr(Loop(a=register, address=Reference(label=label))),
    ]

    return block, LongWaits(n=state.n + 1)


long_wait = LineRule[LongWaits](match=_match_long_wait, map=_map_long_wait)
"""Split a statically long wait.

It accounts for the wait instruction limit, defined by :const:`MAX_WAIT`.

``n`` is used for labelling the loop, and it should be different for each wait
instruction in a sequencer.
"""


def _start_merge_wait(line: Line, state: None) -> tuple[bool, None]:
    # TODO:
    return not isinstance(line, Line), None


def _end_merge_wait(line: Line, state: None) -> tuple[bool, None]:
    # TODO:
    return not isinstance(line, Line), None


def _merge_wait(block: Block, state: None) -> tuple[Block, None]:
    # TODO:
    return block, None


merge_wait = BlockRule[None](
    initial=_start_merge_wait, final=_end_merge_wait, map=_merge_wait
)
