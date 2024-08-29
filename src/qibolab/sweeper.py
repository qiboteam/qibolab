from enum import Enum, auto
from typing import Optional

import numpy as np
import numpy.typing as npt
from pydantic import model_validator

from .identifier import ChannelId
from .pulses import Pulse
from .serialize import Model


class Parameter(Enum):
    """Sweeping parameters."""

    frequency = auto()
    amplitude = auto()
    duration = auto()
    duration_interpolated = auto()
    relative_phase = auto()
    offset = auto()


ChannelParameter = {
    Parameter.frequency,
    Parameter.offset,
}


class Sweeper(Model):
    """Data structure for Sweeper object.

    This object is passed as an argument to the method :func:`qibolab.platforms.platform.Platform.execute`
    which enables the user to sweep a specific parameter for one or more pulses. For information on how to
    perform sweeps see :func:`qibolab.platforms.platform.Platform.execute`.

    Example:
        .. testcode::

            import numpy as np
            from qibolab.dummy import create_dummy
            from qibolab.sweeper import Sweeper, Parameter
            from qibolab.sequence import PulseSequence
            from qibolab import ExecutionParameters


            platform = create_dummy()
            qubit = platform.qubits[0]
            natives = platform.natives.single_qubit[0]
            sequence = natives.MZ.create_sequence()
            parameter_range = np.random.randint(10, size=10)
            sweeper = Sweeper(
                parameter=Parameter.frequency, values=parameter_range, channels=[qubit.probe.name]
            )
            platform.execute([sequence], ExecutionParameters(), [[sweeper]])

    Args:
        parameter: parameter to be swept, possible choices are frequency, attenuation, amplitude, current and gain.
        values: array of parameter values to sweep over.
        range: tuple of ``(start, stop, step)`` to sweep over the array ``np.arange(start, stop, step)``.
            Can be provided instead of ``values`` for more efficient sweeps on some instruments.
        pulses : list of `qibolab.pulses.Pulse` to be swept.
        channels: list of channel names for which the parameter should be swept.
    """

    parameter: Parameter
    values: Optional[npt.NDArray] = None
    range: Optional[tuple[float, float, float]] = None
    pulses: Optional[list[Pulse]] = None
    channels: Optional[list[ChannelId]] = None

    @model_validator(mode="after")
    def check_values(self):
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
        if self.range is not None and self.values is not None:
            raise ValueError("'range' and 'values' are mutually exclusive.")
        if self.range is None and self.values is None:
            raise ValueError("Either 'range' or 'values' needs to be provided.")

        if self.range is not None:
            object.__setattr__(self, "values", np.arange(*self.range))

        return self


ParallelSweepers = list[Sweeper]
"""Sweepers that should be iterated in parallel."""
