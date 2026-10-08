from enum import Enum

import numpy as np

from qibolab._core.sweeper import Parameter

from ..q1asm.ast_ import Instruction, Line, Lineable, Move, Register

__all__ = []


class Registers(Enum):
    """Pre-assigned register numbers."""

    bin = Register(number=0)
    bin_reset = Register(number=1)
    shots = Register(number=2)
    wait = Register(number=3)
    phase_delta = Register(number=4)
    zero = Register(number=5)

    @classmethod
    def first_available(cls) -> int:
        return max(r.value.number for r in cls) + 1

    @classmethod
    def init_zero_registers(cls) -> list[Line]:
        """Generate `Move` instructions to initialize registers with zero value."""
        init_specs = [
            (cls.bin, "init bin counter"),
            (cls.bin_reset, "init bin reset"),
            (cls.phase_delta, "init delta phase register"),
            (cls.zero, "init zero register"),
        ]
        return [
            Line(
                instruction=Move(source=0, destination=reg.value),
                comment=comment,
            )
            for reg, comment in init_specs
        ]


def label(line: Lineable, label: str) -> Line:
    return (
        Line(instruction=line, label=label)
        if isinstance(line, Instruction)
        else Line(instruction=line.instruction, comment=line.comment, label=label)
    )


MAX_PARAM = {
    Parameter.amplitude: 2**15 - 1,
    Parameter.offset: 2**15 - 1,
    Parameter.phase: 1e9,
    Parameter.frequency: 2e9,
}
"""Maximum parameter magnitudes in register units.

Declared in https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html#q1-instructions

Ranges may be one-sided (just positive) or two-sided. This is accounted for in
:func:`convert`.
"""

_MAX_PHYSICAL = {
    Parameter.amplitude: 1.0,
    Parameter.offset: 1.0,
    Parameter.phase: 2 * np.pi,
    Parameter.frequency: 500e6,
}
"""Maximum parameters value allowed in physical units.

This is supplementary to :const:`MAX_PARAM`, coming from the same source. It is required
for the conversions and validations, since in Qibolab the parametes are expressed in
physical units, but they need to be converted in register units to upload to Qblox.
"""


def _validate_frequency(frequency: float) -> None:
    """Validates that frequency is within the valid range."""
    max_ = _MAX_PHYSICAL[Parameter.frequency]
    if abs(frequency) >= max_:
        raise ValueError(
            "IF frequency must be a float between "
            f"-{max_} and {max_}. Received: {frequency}"
        )


def _convert_frequency(frequency: float) -> float:
    """Converts frequency values to the encoding used in qblox FPGAs."""
    # TODO: move validation closer to user input
    _validate_frequency(frequency)
    conversion = MAX_PARAM[Parameter.frequency] / _MAX_PHYSICAL[Parameter.frequency]
    return frequency * conversion


def _validate_offset(offset: float) -> None:
    """Validates that offset is within the valid range."""
    max_ = _MAX_PHYSICAL[Parameter.offset]
    if abs(offset) >= max_:
        raise ValueError(
            f"Offset must be a float between -{max_} and {max_}. Received: {offset}"
        )


def _convert_offset(offset: float) -> float:
    """Converts offset values to the encoding used in qblox FPGAs."""
    _validate_offset(offset)
    return np.floor(offset * MAX_PARAM[Parameter.offset])


def _validate_amplitude(amplitude: float) -> None:
    """Validate that amplitude is within the valid range."""
    max_ = _MAX_PHYSICAL[Parameter.amplitude]
    if not abs(amplitude) <= max_:
        raise ValueError(
            f"Amplitude must be a float between -{max_} and {max_}. Received: {amplitude}"
        )


def _convert_amplitude(amplitude: float) -> float:
    """Convert amplitude to the encoding used in Qblox FPGAs."""
    _validate_amplitude(amplitude)
    return amplitude * MAX_PARAM[Parameter.amplitude]


def _validate_phase(phase: float) -> None:
    """Validate phase before wrapping it to a single turn."""
    if not np.isfinite(phase):
        raise ValueError(f"Phase must be finite. Received: {phase}")


def _convert_phase(phase: float) -> float:
    """Convert phase to the encoding used in Qblox FPGAs."""
    _validate_phase(phase)
    return (phase / _MAX_PHYSICAL[Parameter.phase]) % 1.0 * MAX_PARAM[Parameter.phase]


def _validate_sweeper_value(value: float, kind: Parameter) -> None:
    """Validates sweeper value without performing conversion."""
    if kind is Parameter.frequency:
        _validate_frequency(value)
    elif kind is Parameter.offset:
        _validate_offset(value)
    elif kind is Parameter.amplitude:
        _validate_amplitude(value)
    elif kind in (Parameter.relative_phase, Parameter.phase):
        _validate_phase(value)


def convert(value: float, kind: Parameter) -> float:
    """Convert sweeper value in assembly units."""
    if kind is Parameter.amplitude:
        return _convert_amplitude(value)
    if kind in (Parameter.relative_phase, Parameter.phase):
        return _convert_phase(value)
    if kind is Parameter.frequency:
        return _convert_frequency(value) % (2**32)
    if kind is Parameter.offset:
        return _convert_offset(value) % (2**32)
    if kind is Parameter.duration:
        return value
    raise ValueError(f"Unsupported sweeper: {kind.name}")
