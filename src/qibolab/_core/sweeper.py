from enum import Enum, auto
from functools import cache
from typing import Any, Collection, Optional

import numpy as np
import numpy.typing as npt
from pydantic import model_validator

from .components.configs import OscillatorConfig
from .identifier import ChannelId
from .pulses import PulseLike, VirtualZ
from .serialize import Model, eq

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


Range = tuple[float, float, float]


class Sweeper(Model):
    """Data structure for Sweeper object.

    This object is passed as an argument to the method :func:`qibolab.Platform.execute`
    which enables the user to sweep a specific parameter for one or more pulses. For information on how to
    perform sweeps see :func:`qibolab.Platform.execute`.

    Example:
        .. testcode::

            import numpy as np
            from qibolab import Parameter, PulseSequence, Sweeper
            from qibolab.instruments.dummy import create_dummy

            platform = create_dummy()
            qubit = platform.qubits[0]
            natives = platform.natives.single_qubit[0]
            sequence = natives.MZ.create_sequence()
            parameter_range = np.random.randint(10, size=10)
            sweeper = Sweeper(
                parameter=Parameter.frequency, values=parameter_range, channels=[qubit.probe]
            )
            platform.execute([sequence], [[sweeper]])
    """

    parameter: Parameter
    """Parameter to be swept."""
    values: Optional[npt.NDArray] = None
    """Array of parameter values to sweep over."""
    range: Optional[Range] = None
    """Tuple of ``(start, stop, step)``.

    To sweep over the array ``np.arange(start, stop, step)``.
    Can be provided instead of ``values`` for more efficient sweeps on some instruments.
    """
    pulses: Optional[list[PulseLike]] = None
    """List of `qibolab.Pulse` to be swept."""
    channels: Optional[list[ChannelId]] = None
    """List of channel names for which the parameter should be swept."""

    @model_validator(mode="after")
    def check_values(self):
        _alternative_fields((self.pulses, "pulses"), (self.channels, "channels"))

        if self.values is None and self.range is None:
            raise ValueError("Cannot create a sweeper without values or range.")
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

    @property
    def irange(self) -> tuple[float, float, float]:
        """Inferred range.

        Always ensure a range, inferring it from :attr:`values` if :attr:`range` is
        not set.
        """
        if self.range is not None:
            return self.range
        assert self.values is not None
        step = self.values[1] - self.values[0]
        return (self.values[0], self.values[-1] + step, step)

    def __len__(self) -> int:
        """Compute number of iterations."""
        if self.values is not None:
            return len(self.values)
        assert self.range is not None
        return int((self.range[1] - self.range[0]) // self.range[2] + 1)

    def __eq__(self, other: "Sweeper") -> bool:
        """Compare sweepers.

        The comparison requires adaption, since it may involve NumPy arrays, which do
        not generate a single boolean as output of the comparison operator.
        """
        return eq(self, other)

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
) -> dict[PulseLike, Sweeper]:
    """Associate pulses swept to sweepers.

    Essentially, it produces a reverse index from `sweepers`.

    If `parameters` is passed, it limits the selection to pulses whose parameter swept
    is among those listed. By default, all swept pulses are returned.
    """
    # TODO: this is assuming a pulse is only swept by a single sweeper. Which is not
    # always the case. A list of `Sweeper` objects should be returned instead
    return {
        p: sweep
        for parsweep in sweepers
        for sweep in parsweep
        if sweep.parameter in parameters and sweep.pulses is not None
        for p in sweep.pulses
    }


def swept_channels(sweepers: list[ParallelSweepers]) -> set[ChannelId]:
    """Identify channels involved in a sweeper suite."""
    return {
        channel
        for ps in sweepers
        for sweeper in ps
        for channel in (sweeper.channels if sweeper.channels is not None else [])
    }


def _split_sweeper(sweeper: Sweeper) -> ParallelSweepers:
    return (
        [sweeper.model_copy(update={"pulses": [pulse]}) for pulse in sweeper.pulses]
        if sweeper.pulses is not None
        else [sweeper.model_copy(update={"channels": [ch]}) for ch in sweeper.channels]
    )


def _split_sweepers(sweepers: list[ParallelSweepers]) -> list[ParallelSweepers]:
    return [
        [s for sweep in parsweep for s in _split_sweeper(sweep)]
        for parsweep in sweepers
    ]


def _lo_frequency(lo: Optional[OscillatorConfig]) -> float:
    return lo.frequency if lo is not None else 0.0


def _subtract_lo(
    sweepers: list[ParallelSweepers], los: dict[ChannelId, OscillatorConfig]
) -> list[ParallelSweepers]:
    return [
        [
            (sweep - _lo_frequency(los.get(sweep.channels[0])))
            if sweep.parameter is Parameter.frequency
            else sweep
            for sweep in parsweep
        ]
        for parsweep in sweepers
    ]


def _subtract_offset(
    sweepers: list[ParallelSweepers], offsets: dict[ChannelId, float]
) -> list[ParallelSweepers]:
    return [
        [
            (sweep - offsets.get(sweep.channels[0], 0.0))
            if sweep.parameter is Parameter.offset
            else sweep
            for sweep in parsweep
        ]
        for parsweep in sweepers
    ]


def normalize_sweepers(
    sweepers: list[ParallelSweepers],
    los: Optional[dict[ChannelId, OscillatorConfig]] = None,
    offsets: Optional[dict[ChannelId, float]] = None,
) -> list[ParallelSweepers]:
    sweepers = _split_sweepers(sweepers)
    sweepers = _subtract_lo(sweepers, los) if los is not None else sweepers
    return _subtract_offset(sweepers, offsets) if offsets is not None else sweepers
