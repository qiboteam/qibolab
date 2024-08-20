import math
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibo.config import raise_error
from qm import qua
from qm.qua import declare, fixed
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


def check_max_offset(offset: Optional[float], max_offset: float = MAX_OFFSET):
    """Checks if a given offset value exceeds the maximum supported offset.

    This is to avoid sending high currents that could damage lab
    equipment such as amplifiers.
    """
    if max_offset is not None and abs(offset) > max_offset:
        raise_error(
            ValueError, f"{offset} exceeds the maximum allowed offset {max_offset}."
        )


# def _update_baked_pulses(sweeper, qmsequence, config):
#    """Updates baked pulse if duration sweeper is used."""
#    qmpulse = qmsequence.pulse_to_qmpulse[sweeper.pulses[0].id]
#    is_baked = isinstance(qmpulse, BakedPulse)
#    for pulse in sweeper.pulses:
#        qmpulse = qmsequence.pulse_to_qmpulse[pulse.id]
#        if isinstance(qmpulse, BakedPulse):
#            if not is_baked:
#                raise_error(
#                    TypeError,
#                    "Duration sweeper cannot contain both baked and not baked pulses.",
#                )
#            values = np.array(sweeper.values).astype(int)
#            qmpulse.bake(config, values)


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
        # convert to IF frequency for readout and drive pulses
        f0 = math.floor(configs[name].frequency - lo_frequency)
        # check if sweep is within the supported bandwidth [-400, 400] MHz
        max_freq = maximum_sweep_value(values, f0)
        if max_freq > 4e8:
            raise_error(
                ValueError,
                f"Frequency {max_freq} for channel {name} is beyond instrument bandwidth.",
            )
        qua.update_frequency(name, variable + f0)


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
        # if isinstance(instruction, Bake):
        #    instructions.update_kwargs(instruction, amplitude=a)
        # else:
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


def _bias(
    channels: list[Channel],
    values: npt.NDArray,
    variable: _Variable,
    configs: dict[str, Config],
    args: ExecutionArguments,
):
    for channel in channels:
        name = str(channel.name)
        offset = configs[name].offset
        max_value = maximum_sweep_value(values, offset)
        check_max_offset(max_value, MAX_OFFSET)
        b0 = declare(fixed, value=offset)
        with qua.if_((variable + b0) >= 0.49):
            qua.set_dc_offset(f"flux{name}", "single", 0.49)
        with qua.elif_((variable + b0) <= -0.49):
            qua.set_dc_offset(f"flux{name}", "single", -0.49)
        with qua.else_():
            qua.set_dc_offset(f"flux{name}", "single", (variable + b0))


def _duration(
    pulses: list[Pulse],
    values: npt.NDArray,
    variable: _Variable,
    configs: dict[str, Config],
    args: ExecutionArguments,
):
    # TODO: Handle baked pulses
    for pulse in pulses:
        args.parameters[operation(pulse)].duration = variable


INT_TYPE = {Parameter.frequency, Parameter.duration}
"""Sweeper parameters for which we need ``int`` variable type.

The rest parameters need ``fixed`` type.
"""

NORMALIZERS = {
    Parameter.relative_phase: lambda values: values / (2 * np.pi),
    Parameter.duration: lambda values: (values // 4).astype(int),
}
"""Functions to normalize sweeper values.

The rest parameters do not need normalization (identity function).
"""

SWEEPER_METHODS = {
    Parameter.frequency: _frequency,
    Parameter.amplitude: _amplitude,
    Parameter.duration: _duration,
    Parameter.relative_phase: _relative_phase,
    Parameter.bias: _bias,
}
"""Methods that return part of QUA program to be used inside the loop."""
