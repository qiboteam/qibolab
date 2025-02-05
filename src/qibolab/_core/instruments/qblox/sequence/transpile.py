from collections.abc import Sequence
from typing import Union

from ..q1asm.ast_ import (
    Instruction,
    Line,
    Loop,
    Move,
    Program,
    Reference,
    Register,
    Wait,
)

__all__ = []

MAX_WAIT = 2**16 - 1

Block = Sequence[Union[Line, Instruction]]


def long_wait(duration: int) -> Block:
    n = 60
    iterations = duration // MAX_WAIT
    remainder = duration % MAX_WAIT
    register = Register(number=n)
    label = f"wait-{n}"
    return [Wait(duration=remainder)] + [
        Move(source=iterations, destination=register),
        Line(instruction=Wait(duration=MAX_WAIT), label=label),
        Line.instr(Loop(a=register, address=Reference(label=label))),
    ]


def decompose(line: Line) -> Block:
    if not isinstance(line.instruction, Wait):
        return [line]
    duration = line.instruction.duration
    if not isinstance(duration, int) or duration <= MAX_WAIT:
        return [line]
    wait = long_wait(duration)
    assert isinstance(wait[0], Instruction)
    return [
        Line(instruction=wait[0], label=line.label, comment=line.comment),
        *(el for el in wait[1:]),
    ]


def transpile(prog: Block) -> Program:
    return Program(
        elements=[
            el if isinstance(el, Line) else Line.instr(el)
            for oel in prog
            for el in (decompose(oel) if isinstance(oel, Line) else [oel])
        ]
    )
