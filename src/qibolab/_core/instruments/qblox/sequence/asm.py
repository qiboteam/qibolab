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


def convert_offset(offset: float):
    """Converts offset values to the encoding used in qblox FPGAs.

    Both offset values are divided in 2**sample path width steps. QCM
    DACs resolution 16bits, QRM DACs and ADCs 12 bit QCM 5Vpp, QRM 2Vpp

    https://docs.qblox.com/en/main/cluster/qcm.html#specifications
    https://docs.qblox.com/en/main/cluster/qcm_rf.html#specifications
    https://docs.qblox.com/en/main/cluster/qrm_rf.html#specifications
    """
    # FIXME: Copied from Qibolab 0.1 - no idea where the conversion is coming from
    scale_factor = 1.25 * np.sqrt(2)
    normalised_offset = offset / scale_factor

    # TODO: move validation closer to user input
    if (normalised_offset <= -1) or (1 <= normalised_offset):
        raise ValueError(
            f"offset must be a float between {-scale_factor:.3f} and {scale_factor:.3f} V"
        )

    max = MAX_PARAM[Parameter.offset]
    mapped = np.floor(normalised_offset * (max + 1)).astype(int)
    # clipping required to avoid wrapping offset == 1 to a negative value
    return np.maximum(mapped, max)


def convert(value: float, kind: Parameter) -> float:
    """Convert sweeper value in assembly units."""
    if kind is Parameter.amplitude:
        return value * MAX_PARAM[kind]
    if kind is Parameter.relative_phase:
        # TODO: the following is actually redundant, choose what to keep
        # most likely the maximum value, set to 1e9, is something like 2**30 (not sure
        # why not 2**32), and the three % operations are all doing the same
        return ((value % (2 * np.pi)) / (2 * np.pi)) % 1.0 * MAX_PARAM[kind] % (2**32)
    if kind is Parameter.frequency:
        return 4 * value % (2**32)
    if kind is Parameter.offset:
        return convert_offset(value) % (2**32)
    if kind is Parameter.duration:
        return value
    raise ValueError(f"Unsupported sweeper: {kind.name}")
