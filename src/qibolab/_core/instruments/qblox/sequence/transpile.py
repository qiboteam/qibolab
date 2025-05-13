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


def _long_wait(duration: int) -> Block:
    iterations = duration // MAX_WAIT
    remainder = duration % MAX_WAIT
    register = Registers.wait.value
    label = f"wait{register.number}"
    return [Wait(duration=remainder)] + [
        Move(source=iterations, destination=register),
        Line(instruction=Wait(duration=MAX_WAIT), label=label),
        Line.instr(Loop(a=register, address=Reference(label=label))),
    ]


def _decompose_wait(instr: Wait) -> Optional[Block]:
    duration = instr.duration
    if not isinstance(duration, int) or duration <= MAX_WAIT:
        return None
    return _long_wait(duration)


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


def _line_transform(line: Line) -> list[Line]:
    instr = line.instruction
    block = (
        _decompose_wait(instr)
        if isinstance(instr, Wait)
        else _decompose_move(instr)
        if isinstance(instr, Move)
        else None
    )

    # default
    if block is None:
        return [line]

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
    ]


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


def _merge_wait(prog: list[Line]) -> list[Line]:
    batch = _WaitBatch()
    new = []
    for line in prog:
        instr = line.instruction
        intwait = isinstance(instr, Wait) and isinstance(instr.duration, int)
        if not intwait or line.label is not None:
            new += batch.lines + ([line] if not intwait else [])
            batch = _WaitBatch()
        if intwait:
            batch.increment(line)

    return new + batch.lines


def _block_transform(prog: Block) -> list[Line]:
    lines = [el if isinstance(el, Line) else Line.instr(el) for el in prog]
    return _merge_wait(lines)


def transpile(prog: Block) -> Program:
    return Program(
        elements=[el for oel in _block_transform(prog) for el in _line_transform(oel)]
    )
