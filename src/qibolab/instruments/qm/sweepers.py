import math
from dataclasses import dataclass

import numpy as np
from qibo.config import raise_error
from qm import qua
from qm.qua import declare, fixed, for_
from qualang_tools.loops import from_array

from qibolab.components import Config
from qibolab.sweeper import Parameter, Sweeper

from .config import operation
from .program import ExecutionArguments, play

MAX_OFFSET = 0.5
"""Maximum voltage supported by Quantum Machines OPX+ instrument in volts."""


def maximum_sweep_value(values, value0):
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


def check_max_offset(offset, max_offset=MAX_OFFSET):
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


@dataclass
class QuaSweep:

    sweeper: Sweeper

    def declare(self):
        return declare(fixed)

    @property
    def values(self):
        return self.sweeper.values

    def __call__(self, variable, configs, args):
        raise NotImplementedError


class Frequency(QuaSweep):

    def declare(self):
        return declare(int)

    def __call__(self, variable, configs, args):
        for channel in self.sweeper.channels:
            lo_frequency = configs[channel.lo].frequency
            # convert to IF frequency for readout and drive pulses
            f0 = math.floor(configs[channel.name].frequency - lo_frequency)
            # check if sweep is within the supported bandwidth [-400, 400] MHz
            max_freq = maximum_sweep_value(self.values, f0)
            if max_freq > 4e8:
                raise_error(
                    ValueError,
                    f"Frequency {max_freq} for channel {channel.name} is beyond instrument bandwidth.",
                )
            qua.update_frequency(channel.name, variable + f0)


class Amplitude(QuaSweep):

    def __call__(self, variable, configs, args):
        # TODO: Consider sweeping amplitude without multiplication
        if min(self.values) < -2:
            raise_error(
                ValueError, "Amplitude sweep values are <-2 which is not supported."
            )
        if max(self.values) > 2:
            raise_error(
                ValueError, "Amplitude sweep values are >2 which is not supported."
            )

        for pulse in self.sweeper.pulses:
            # if isinstance(instruction, Bake):
            #    instructions.update_kwargs(instruction, amplitude=a)
            # else:
            args.parameters[operation(pulse)].amplitude = qua.amp(variable)


class RelativePhase(QuaSweep):

    @property
    def values(self):
        return self.sweeper.values / (2 * np.pi)

    def __call__(self, variable, configs, args):
        for pulse in self.sweeper.pulses:
            args.parameters[operation(pulse)].phase = variable


class Bias(QuaSweep):

    def __call__(self, variable, configs, args):
        for channel in self.sweeper.channels:
            offset = configs[channel.name].offset
            max_value = maximum_sweep_value(self.values, offset)
            check_max_offset(max_value, MAX_OFFSET)
            b0 = declare(fixed, value=offset)
            with qua.if_((variable + b0) >= 0.49):
                qua.set_dc_offset(f"flux{channel.name}", "single", 0.49)
            with qua.elif_((variable + b0) <= -0.49):
                qua.set_dc_offset(f"flux{channel.name}", "single", -0.49)
            with qua.else_():
                qua.set_dc_offset(f"flux{channel.name}", "single", (variable + b0))


class Duration(QuaSweep):
    def declare(self):
        return declare(int)

    @property
    def values(self):
        return (self.sweeper.values // 4).astype(int)

    def __call__(self, variable, configs, args):
        # TODO: Handle baked pulses
        for pulse in self.sweeper.pulses:
            args.parameters[operation(pulse)].duration = variable


QUA_SWEEPERS = {
    Parameter.frequency: Frequency,
    Parameter.amplitude: Amplitude,
    Parameter.duration: Duration,
    Parameter.relative_phase: RelativePhase,
    Parameter.bias: Bias,
}


def _sweep_recursion(sweepers, configs, args):
    """Unrolls a list of qibolab sweepers to the corresponding QUA for loops
    using recursion."""
    if len(sweepers) > 0:
        sweeper = sweepers[0]
        parameter = sweeper.parameter
        if parameter in QUA_SWEEPERS:
            qua_sweeper = QUA_SWEEPERS[parameter](sweeper)
        else:
            raise_error(
                NotImplementedError, f"Sweeper for {parameter} is not implemented."
            )

        variable = qua_sweeper.declare()
        with for_(*from_array(variable, qua_sweeper.values)):
            qua_sweeper(variable, configs, args)
            _sweep_recursion(sweepers[1:], configs, args)

    else:
        play(args)


def sweep(
    sweepers: list[Sweeper], configs: dict[str, Config], args: ExecutionArguments
):
    """Public sweep function that is called by the driver."""
    # for sweeper in sweepers:
    #    if sweeper.parameter is Parameter.duration:
    #        _update_baked_pulses(sweeper, instructions, config)
    _sweep_recursion(sweepers, configs, args)
