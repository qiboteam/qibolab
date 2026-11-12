from qibolab._core.serialize import Model

from ....q1asm.ast_ import (
    Block,
    Jlt,
    Jmp,
    Line,
    Loop,
    Move,
    Reference,
    Register,
    Sub,
    Wait,
)
from ...asm import Registers
from ..components import BlockRule, LineRule

__all__ = [
    "LongWaits",
    "long_wait",
    "merge_wait",
    "swept_wait",
]

MAX_WAIT = 2**16 - 1
MIN_WAIT = 4
STEP = MAX_WAIT - MIN_WAIT + 1
"""Loop wait, short enough that the remainder always fits :const:`MIN_WAIT`."""


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

    iterations, remainder = divmod(duration - MIN_WAIT, STEP)
    register = Registers.wait.value
    label = f"wait{state.n}"

    block = [
        Wait(duration=remainder + MIN_WAIT),
        Move(source=iterations, destination=register),
        Line(instruction=Wait(duration=STEP), label=label),
        Line.instr(Loop(a=register, address=Reference(label=label))),
    ]

    return block, LongWaits(n=state.n + 1)


long_wait = LineRule(init_state=LongWaits, match=_match_long_wait, map=_map_long_wait)
"""Split a long immediate wait.

It accounts for the wait instruction limit, defined by :const:`MAX_WAIT`.

Each loop is labelled with its own unique tag, which is assigned sequentially by keeping
the count of expanded loops.
"""


def _match_swept_wait(line: Line, state: LongWaits) -> tuple[bool, LongWaits]:
    instr = line.instruction
    return isinstance(instr, Wait) and isinstance(instr.duration, Register), state


def _map_swept_wait(line: Line, state: LongWaits) -> tuple[Block, LongWaits]:
    instr = line.instruction
    assert isinstance(instr, Wait) and isinstance(instr.duration, Register)

    register = Registers.wait.value
    loop, end = f"swept_wait{state.n}", f"swept_wait_end{state.n}"

    block = [
        Move(source=instr.duration, destination=register),
        Line(
            instruction=Jlt(a=register, b=MAX_WAIT + 1, address=Reference(label=end)),
            label=loop,
        ),
        Wait(duration=STEP),
        Sub(a=register, b=STEP, destination=register),
        Jmp(address=Reference(label=loop)),
        Line(instruction=Wait(duration=register), label=end),
    ]

    return block, LongWaits(n=state.n + 1)


swept_wait = LineRule(
    init_state=LongWaits, match=_match_swept_wait, map=_map_swept_wait
)
"""Split a register wait at runtime.

``wait`` only reads the lower 16 bits of a register, so its value is consumed in chunks
until the remainder fits :const:`MAX_WAIT`. The register is left untouched.
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


merge_wait = BlockRule(
    init_state=lambda: None,
    start=_start_merge_wait,
    end=_end_merge_wait,
    map=_merge_wait,
)
