from enum import Enum

import numpy as np

from qibolab._core.sweeper import Parameter

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


MAX_PARAM = {
    Parameter.amplitude: 2**15 - 1,
    Parameter.offset: 2**15 - 1,
    Parameter.relative_phase: 1e9,
    Parameter.frequency: 2e9,
}
"""Maximum range for parameters.

Declared in https://docs.qblox.com/en/main/cluster/q1_sequence_processor.html#q1-instructions

Ranges may be one-sided (just positive) or two-sided. This is accounted for in
:func:`convert`.
"""


def convert(value: float, kind: Parameter) -> float:
    """Convert sweeper value in assembly units."""
    if kind is Parameter.amplitude:
        return value * MAX_PARAM[kind]
    if kind is Parameter.relative_phase:
        return ((value % (2 * np.pi)) / (2 * np.pi)) % 1.0 * MAX_PARAM[kind]
    if kind is Parameter.frequency:
        # return value / 500e6 * MAX_PARAM[kind]
        return int(4 * value) % (2**32)
    if kind is Parameter.offset:
        return value * MAX_PARAM[kind]
    if kind is Parameter.duration:
        return value
    raise ValueError(f"Unsupported sweeper: {kind.name}")
