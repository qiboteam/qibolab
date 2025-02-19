from enum import Enum, auto
from functools import cache, cached_property
from typing import Any, Collection, Container, Optional

import numpy as np
import numpy.typing as npt
from pydantic import model_validator

from qibolab._core.pulses.pulse import PulseId

from .identifier import ChannelId
from .pulses import PulseLike, VirtualZ
from .serialize import Model

__all__ = ["Parameter", "ParallelSweepers", "Sweeper"]

_PULSE = "pulse"
_CHANNEL = "channel"


class Parameter(Enum):
    """Sweeping parameters."""

    frequency = (auto(), _CHANNEL)
    amplitude = (auto(), _PULSE)
    duration = (auto(), _PULSE)
    duration_interpolated = (auto(), _PULSE)
    relative_phase = (auto(), _PULSE)
    phase = (auto(), _PULSE)
    offset = (auto(), _CHANNEL)

    @classmethod
    @cache
    def channels(cls) -> set["Parameter"]:
        """Set of parameters to be swept on the channel."""
        return {p for p in cls if p.value[1] == _CHANNEL}


_Field = tuple[Any, str]


def _alternative_fields(a: _Field, b: _Field):
    if (a[0] is None) == (b[0] is None):
        raise ValueError(
            f"Either '{a[1]}' or '{b[1]}' needs to be provided, and only one of them."
        )


class Sweeper(Model):
    """Data structure for Sweeper object.

    This object is passed as an argument to the method :func:`qibolab.Platform.execute`
    which enables the user to sweep a specific parameter for one or more pulses. For information on how to
    perform sweeps see :func:`qibolab.Platform.execute`.

    Example:
        .. testcode::

            import numpy as np
            from qibolab import Parameter, PulseSequence, Sweeper, create_dummy


            platform = create_dummy()
            qubit = platform.qubits[0]
            natives = platform.natives.single_qubit[0]
            sequence = natives.MZ.create_sequence()
            parameter_range = np.random.randint(10, size=10)
            sweeper = Sweeper(
                parameter=Parameter.frequency, values=parameter_range, channels=[qubit.probe]
            )
            platform.execute([sequence], [[sweeper]])

    Args:
        parameter: parameter to be swept, possible choices are frequency, attenuation, amplitude, current and gain.
        values: array of parameter values to sweep over.
        range: tuple of ``(start, stop, step)`` to sweep over the array ``np.arange(start, stop, step)``.
            Can be provided instead of ``values`` for more efficient sweeps on some instruments.
        pulses : list of `qibolab.Pulse` to be swept.
        channels: list of channel names for which the parameter should be swept.
    """

    parameter: Parameter
    values: Optional[npt.NDArray] = None
    range: Optional[tuple[float, float, float]] = None
    pulses: Optional[list[PulseLike]] = None
    channels: Optional[list[ChannelId]] = None

    @model_validator(mode="after")
    def check_values(self):
        _alternative_fields((self.pulses, "pulses"), (self.channels, "channels"))
        _alternative_fields((self.range, "range"), (self.values, "values"))

        if self.pulses is not None and self.parameter in Parameter.channels():
            raise ValueError(
                f"Cannot create a sweeper for {self.parameter} without specifying channels."
            )
        if self.parameter not in Parameter.channels() and (self.channels is not None):
            raise ValueError(
                f"Cannot create a sweeper for {self.parameter} without specifying pulses."
            )
        if self.parameter is Parameter.phase and not all(
            isinstance(pulse, VirtualZ) for pulse in self.pulses
        ):
            raise TypeError("Cannot create a phase sweeper on non-VirtualZ pulses.")

        if self.range is not None:
            object.__setattr__(self, "values", np.arange(*self.range))

        if self.parameter is Parameter.amplitude and max(abs(self.values)) > 1:
            raise ValueError(
                "Amplitude sweeper cannot have absolute values larger than 1."
            )

        return self

    @cached_property
    def irange(self) -> tuple[float, float, float]:
        """Inferred range.

        Always ensure a range, inferring it from :attr:`values` if :attr:`range` is
        not set.
        """
        if self.range is not None:
            return self.range
        assert self.values is not None
        return (self.values[0], self.values[-1], self.values[1] - self.values[0])

    def __len__(self) -> int:
        """Compute number of iterations."""
        if self.values is not None:
            return len(self.values)
        assert self.range is not None
        return int((self.range[1] - self.range[0]) // self.range[2] + 1)

    def __add__(self, value: float) -> "Sweeper":
        """Add value to sweeper ones."""
        return self.model_copy(
            update=(
                {"range": (self.range[0] + value, self.range[1] + value, self.range[2])}
                if self.range is not None
                else {}
            )
            | ({"values": self.values + value} if self.values is not None else {})
        )

    def __sub__(self, value: float) -> "Sweeper":
        """Subtract value from sweeper ones."""
        return self + (-value)

    def __mul__(self, value: float) -> "Sweeper":
        """Multiply value to sweeper ones.

        TODO: deduplicate this and :meth:`__add__`
        """
        return self.model_copy(
            update=(
                {"range": (self.range[0] * value, self.range[1] * value, self.range[2])}
                if self.range is not None
                else {}
            )
            | ({"values": self.values * value} if self.values is not None else {})
        )

    def __truediv__(self, value: float) -> "Sweeper":
        """Divide by value from sweeper ones."""
        return self * (1 / value)


ParallelSweepers = list[Sweeper]
"""Sweepers that should be iterated in parallel."""


def iteration_length(sweepers: ParallelSweepers) -> int:
    """Compute lenght of parallel iteration."""
    return min((len(s) for s in sweepers), default=0)


def swept_pulses(
    sweepers: list[ParallelSweepers],
    parameters: Collection[Parameter] = frozenset(Parameter),
) -> set[PulseId]:
    """Extract identifiers of swept pulses.

    If `parameters` is passed, it limits the selection to pulses whose parameter swept
    is among those listed. By default, all swept pulses are returned.
    """
    return {
        p.id
        for parsweep in sweepers
        for sweep in parsweep
        if sweep.parameter in parameters and sweep.pulses is not None
        for p in sweep.pulses
    }
