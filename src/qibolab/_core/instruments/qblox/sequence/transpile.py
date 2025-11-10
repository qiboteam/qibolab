from functools import reduce
from typing import Optional, cast

from pydantic import BaseModel, Field

from ..q1asm.ast_ import (
    Block,
    Instruction,
    Line,
    Loop,
    Move,
    Program,
    Reference,
    Register,
    Sub,
    Wait,
)
from .asm import Registers

__all__ = []

MAX_WAIT = 2**16 - 1


class State(BaseModel):
    nwait: int = 0


def _long_wait(duration: int, n: int) -> Block:
    """Split a statically long wait.

    It accounts for the wait instruction limit, defined by :const:`MAX_WAIT`.

    ``n`` is used for labelling the loop, and it should be different for each wait
    instruction in a sequencer.
    """
    iterations = duration // MAX_WAIT
    remainder = duration % MAX_WAIT
    register = Registers.wait.value
    label = f"wait{register.number}"
    return [Wait(duration=remainder)] + [
        Move(source=iterations, destination=register),
        Line(instruction=Wait(duration=MAX_WAIT), label=label),
        Line.instr(Loop(a=register, address=Reference(label=label))),
    ]


def _decompose_wait(instr: Wait, n: int) -> Optional[Block]:
    duration = instr.duration
    if not isinstance(duration, int) or duration <= MAX_WAIT:
        return None
    return _long_wait(duration, n)


def _negative_move(instr: Move):
    """Compile negative value sets.

    Apparently, the only place where negative numbers are not allowed
    are registers, otherwise they are handled by the internal compiler.

    https://docs.qblox.com/en/main/tutorials/q1asm_tutorials/intermediate/nco_control_adv.html#:~:text=Internally,%20the%20processor%20stores

    Thus, we compile instructions setting negative values as suggested:
    first setting them to 0, than subtracting the desired amount. This
    is more reliable than manually complementing the number, since it
    makes no assumption about the registers size.

    https://docs.qblox.com/en/main/cluster/troubleshooting.html#:~:text=How%20do%20I%20set%20negative%20numbers
    """
    src = cast(int, instr.source)
    dest = instr.destination
    return [Move(source=0, destination=dest), Sub(a=dest, b=abs(src), destination=dest)]


def _decompose_move(instr: Move) -> Optional[Block]:
    src = instr.source
    if isinstance(src, Register):
        return None
    assert isinstance(src, int)
    if src >= 0:
        return None
    return _negative_move(instr)


LineTransformed = list[Line]


def _line_transform(line: Line, state: State) -> tuple[LineTransformed, State]:
    instr = line.instruction
    block, state = (
        (
            _decompose_wait(instr, state.nwait),
            state.model_copy(update={"nwait": state.nwait + 1}),
        )
        if isinstance(instr, Wait)
        else (_decompose_move(instr), state)
        if isinstance(instr, Move)
        else (None, state)
    )

    # default
    if block is None:
        return [line], state

    assert isinstance(block[0], Instruction)
    return [
        el if isinstance(el, Line) else Line.instr(el)
        for el in (
            (
                Line(instruction=block[0], label=line.label, comment=line.comment),
                *(el for el in block[1:]),
            )
            if block is not None
            else [line]
        )
    ], state


def _line_transform_apply(
    value: tuple[list[LineTransformed], State], line: Line
) -> tuple[list[LineTransformed], State]:
    """Accumulate."""
    transformed, state = _line_transform(line, value[1])
    return (value[0] + [transformed]), state


class _WaitBatch(BaseModel):
    duration: int = 0
    comment: list[str] = Field(default_factory=list)
    label: Optional[str] = None

    def increment(self, line: Line):
        instr = line.instruction
        assert isinstance(instr, Wait) and isinstance(instr.duration, int)
        self.duration += instr.duration
        if line.comment is not None:
            self.comment.append(line.comment)
        if line.label is not None:
            self.label = line.label

    @property
    def lines(self) -> list[Line]:
        return (
            [
                Line(
                    label=self.label,
                    instruction=Wait(duration=self.duration),
                    comment="\n".join(self.comment),
                )
            ]
            if self.duration > 0
            else []
        )


def _merge_wait(block: list[Line]) -> list[Line]:
    """Merge subsequent static (immediate) waits."""
    batch = _WaitBatch()
    new = []
    for line in block:
        instr = line.instruction
        intwait = isinstance(instr, Wait) and isinstance(instr.duration, int)
        if not intwait or line.label is not None:
            new += batch.lines + ([line] if not intwait else [])
            batch = _WaitBatch()
        if intwait:
            batch.increment(line)

    return new + batch.lines


def _block_transform(block: Block) -> list[Line]:
    lines = [el if isinstance(el, Line) else Line.instr(el) for el in block]
    return _merge_wait(lines)


def transpile(prog: Block) -> Program:
    block_replaced = _block_transform(prog)
    lines_replaced, _state = reduce(
        _line_transform_apply, block_replaced, ([], State())
    )
    return Program(elements=[line for block in lines_replaced for line in block])
