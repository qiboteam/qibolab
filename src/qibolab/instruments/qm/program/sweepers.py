import numpy as np
import numpy.typing as npt
from qibo.config import raise_error
from qm import qua
from qm.qua._dsl import _Variable  # for type declaration only

from qibolab.components import Channel, Config
from qibolab.pulses import Pulse
from qibolab.sweeper import Parameter

from ..config import operation
from .arguments import ExecutionArguments

MAX_OFFSET = 0.5
"""Maximum voltage supported by Quantum Machines OPX+ instrument in volts."""


def maximum_sweep_value(values: npt.NDArray, value0: npt.NDArray) -> float:
    """Calculates maximum value that is reached during a sweep.

    Useful to check whether a sweep exceeds the range of allowed values.
    Note that both the array of values we sweep and the center value can
    be negative, so we need to make sure that the maximum absolute value
    is within range.

    Args:
        values (np.ndarray): Array of values we will sweep over.
        value0 (float, int): Center value of the sweep.
    """
    return max(abs(min(values) + value0), abs(max(values) + value0))


def _frequency(
    channels: list[Channel],
    values: npt.NDArray,
    variable: _Variable,
    configs: dict[str, Config],
    args: ExecutionArguments,
):
    for channel in channels:
        name = str(channel.name)
        lo_frequency = configs[channel.lo].frequency
        # check if sweep is within the supported bandwidth [-400, 400] MHz
        max_freq = maximum_sweep_value(values, -lo_frequency)
        if max_freq > 4e8:
            raise_error(
                ValueError,
                f"Frequency {max_freq} for channel {name} is beyond instrument bandwidth.",
            )
        qua.update_frequency(name, variable - lo_frequency)


def _amplitude(
    pulses: list[Pulse],
    values: npt.NDArray,
    variable: _Variable,
    configs: dict[str, Config],
    args: ExecutionArguments,
):
    # TODO: Consider sweeping amplitude without multiplication
    if min(values) < -2:
        raise_error(
            ValueError, "Amplitude sweep values are <-2 which is not supported."
        )
    if max(values) > 2:
        raise_error(ValueError, "Amplitude sweep values are >2 which is not supported.")

    for pulse in pulses:
        args.parameters[operation(pulse)].amplitude = qua.amp(variable)


def _relative_phase(
    pulses: list[Pulse],
    values: npt.NDArray,
    variable: _Variable,
    configs: dict[str, Config],
    args: ExecutionArguments,
):
    for pulse in pulses:
        args.parameters[operation(pulse)].phase = variable


def _offset(
    channels: list[Channel],
    values: npt.NDArray,
    variable: _Variable,
    configs: dict[str, Config],
    args: ExecutionArguments,
):
    for channel in channels:
        name = str(channel.name)
        max_value = maximum_sweep_value(values, 0)
        with qua.if_(variable >= MAX_OFFSET):
            qua.set_dc_offset(name, "single", MAX_OFFSET)
        with qua.elif_(variable <= -MAX_OFFSET):
            qua.set_dc_offset(name, "single", -MAX_OFFSET)
        with qua.else_():
            qua.set_dc_offset(name, "single", variable)


def _duration(
    pulses: list[Pulse],
    values: npt.NDArray,
    variable: _Variable,
    configs: dict[str, Config],
    args: ExecutionArguments,
):
    for pulse in pulses:
        args.parameters[operation(pulse)].duration = variable


def _duration_interpolated(
    pulses: list[Pulse],
    values: npt.NDArray,
    variable: _Variable,
    configs: dict[str, Config],
    args: ExecutionArguments,
):
    for pulse in pulses:
        params = args.parameters[operation(pulse)]
        params.duration = variable
        params.interpolated = True


def normalize_phase(values):
    """Normalize phase from [0, 2pi] to [0, 1]."""
    return values / (2 * np.pi)


def normalize_duration(values):
    """Convert duration from ns to clock cycles (clock cycle = 4ns)."""
    if any(values < 16) and not all(values % 4 == 0):
        raise ValueError(
            "Cannot use interpolated duration sweeper for durations that are not multiple of 4ns or are less than 16ns. Please use normal duration sweeper."
        )
    return (values // 4).astype(int)


INT_TYPE = {Parameter.frequency, Parameter.duration, Parameter.duration_interpolated}
"""Sweeper parameters for which we need ``int`` variable type.

The rest parameters need ``fixed`` type.
"""

NORMALIZERS = {
    Parameter.relative_phase: normalize_phase,
    Parameter.duration_interpolated: normalize_duration,
}
"""Functions to normalize sweeper values.

The rest parameters do not need normalization (identity function).
"""

SWEEPER_METHODS = {
    Parameter.frequency: _frequency,
    Parameter.amplitude: _amplitude,
    Parameter.duration: _duration,
    Parameter.duration_interpolated: _duration_interpolated,
    Parameter.relative_phase: _relative_phase,
    Parameter.offset: _offset,
}
"""Methods that return part of QUA program to be used inside the loop."""
