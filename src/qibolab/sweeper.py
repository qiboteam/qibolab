import operator
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import Optional

import numpy.typing as npt


class Parameter(Enum):
    """Sweeping parameters."""

    frequency = auto()
    amplitude = auto()
    duration = auto()
    relative_phase = auto()

    attenuation = auto()
    gain = auto()
    bias = auto()
    lo_frequency = auto()


FREQUENCY = Parameter.frequency
AMPLITUDE = Parameter.amplitude
DURATION = Parameter.duration
RELATIVE_PHASE = Parameter.relative_phase
ATTENUATION = Parameter.attenuation
GAIN = Parameter.gain
BIAS = Parameter.bias


class SweeperType(Enum):
    """Type of the Sweeper."""

    ABSOLUTE = partial(lambda x, y=None: x)
    FACTOR = operator.mul
    OFFSET = operator.add


ChannelParameter = {
    Parameter.frequency,
    Parameter.bias,
    Parameter.attenuation,
    Parameter.gain,
}


@dataclass
class Sweeper:
    """Data structure for Sweeper object.

    This object is passed as an argument to the method :func:`qibolab.platforms.platform.Platform.execute`
    which enables the user to sweep a specific parameter for one or more pulses. For information on how to
    perform sweeps see :func:`qibolab.platforms.platform.Platform.execute`.

    Example:
        .. testcode::

            import numpy as np
            from qibolab.dummy import create_dummy
            from qibolab.sweeper import Sweeper, Parameter
            from qibolab.pulses import PulseSequence
            from qibolab import ExecutionParameters


            platform = create_dummy()
            qubit = platform.qubits[0]
            natives = platform.parameters.native_gates.single_qubit[0]
            sequence = natives.MZ.create_sequence()
            parameter = Parameter.frequency
            parameter_range = np.random.randint(10, size=10)
            sweeper = Sweeper(parameter, parameter_range, channels=[qubit.probe.name])
            platform.execute([sequence], ExecutionParameters(), [[sweeper]])

    Args:
        parameter: parameter to be swept, possible choices are frequency, attenuation, amplitude, current and gain.
        values: sweep range. If the parameter of the sweep is a pulse parameter, if the sweeper type is not ABSOLUTE, the base value
            will be taken from the runcard pulse parameters. If the sweep parameter is Bias, the base value will be the sweetspot of the qubits.
        pulses : list of `qibolab.pulses.Pulse` to be swept.
        channels: list of channel names for which the parameter should be swept.
        type: can be ABSOLUTE (the sweeper range is swept directly),
            FACTOR (sweeper values are multiplied by base value), OFFSET (sweeper values are added
            to base value)
    """

    parameter: Parameter
    values: npt.NDArray
    pulses: Optional[list] = None
    channels: Optional[list] = None
    type: Optional[SweeperType] = SweeperType.ABSOLUTE

    def __post_init__(self):
        if self.pulses is not None and self.channels is not None:
            raise ValueError(
                "Cannot create a sweeper by using both pulses and channels."
            )
        if self.pulses is not None and self.parameter in ChannelParameter:
            raise ValueError(
                f"Cannot create a sweeper for {self.parameter} without specifying channels."
            )
        if self.parameter not in ChannelParameter and (self.channels is not None):
            raise ValueError(
                f"Cannot create a sweeper for {self.parameter} without specifying pulses."
            )
        if self.pulses is None and self.channels is None:
            raise ValueError(
                "Cannot create a sweeper without specifying pulses or channels."
            )

    def get_values(self, base_value):
        """Convert sweeper values depending on the sweeper type."""
        return self.type.value(self.values, base_value)


ParallelSweepers = list[Sweeper]
"""Sweepers that should be iterated in parallel."""
