from enum import Enum

from ..q1asm.ast_ import Instruction, Line, Lineable, Register

__all__ = []


class Registers(Enum):
    """Pre-assigned register numbers."""

    bin = Register(number=0)
    bin_reset = Register(number=1)
    shots = Register(number=2)
    wait = Register(number=3)

    @classmethod
    def first_available(cls) -> int:
        return max(r.value.number for r in cls) + 1


def label(line: Lineable, label: str) -> Line:
    return (
        Line(instruction=line, label=label)
        if isinstance(line, Instruction)
        else Line(instruction=line.instruction, comment=line.comment, label=label)
    )
