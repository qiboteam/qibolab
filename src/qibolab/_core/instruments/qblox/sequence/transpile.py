from ..q1asm.ast_ import (
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


def long_wait(duration: int) -> list[Line]:
    n = 60
    iterations = duration // MAX_WAIT
    remainder = duration % MAX_WAIT
    register = Register(number=n)
    label = f"wait-{n}"
    return [Line.instr(Wait(duration=remainder))] + [
        Line.instr(Move(source=iterations, destination=register)),
        Line(instruction=Wait(duration=MAX_WAIT), label=label),
        Line.instr(Loop(a=register, address=Reference(label=label))),
    ]


def decompose(line: Line) -> list[Line]:
    if not isinstance(line.instruction, Wait):
        return [line]
    duration = line.instruction.duration
    if not isinstance(duration, int) or duration <= MAX_WAIT:
        return [line]
    wait = long_wait(duration)
    return [
        Line(instruction=wait[0].instruction, label=line.label, comment=line.comment)
    ] + wait[1:]


def transpile(prog: Program) -> Program:
    return Program(
        elements=[
            el
            for oel in prog.elements
            for el in (decompose(oel) if isinstance(oel, Line) else [oel])
        ]
    )
