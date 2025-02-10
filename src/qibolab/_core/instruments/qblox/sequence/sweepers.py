from collections.abc import Callable, Iterable, Sequence
from itertools import groupby
from typing import Optional

import numpy as np

from qibolab._core.identifier import ChannelId
from qibolab._core.instruments.qblox.q1asm.ast_ import SetAwgGain, SetAwgOffs, SetFreq
from qibolab._core.pulses.pulse import (
    Pulse,
    PulseId,
    PulseLike,
)
from qibolab._core.serialize import Model
from qibolab._core.sweeper import ParallelSweepers, Parameter

from ..q1asm.ast_ import (
    Instruction,
    Register,
    SetPhDelta,
    Value,
)

__all__ = []


class Param(Model):
    reg: Register
    """Register used for the parameter value."""
    start: int
    """Initial value."""
    step: int
    """Increment."""
    kind: Parameter
    """The parameter type."""
    sweeper: int
    """The loop to which is associated."""
    pulse: Optional[PulseId]
    """The target pulse (if the sweeper targets pulses)."""
    channel: Optional[ChannelId]
    """The target channel (if the sweeper targets channels)."""

    @property
    def description(self):
        """Textual description, used in some accompanying comments."""
        return f"sweeper {self.sweeper + 1} (pulse: {self.pulse})"


Params = Sequence[tuple[int, Param]]
"""Sequence of update parameters.

It is created by the :func:`params` function.
"""

IndexedParams = dict[int, tuple[list[Param], list[Param]]]

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


def _convert(value: float, kind: Parameter) -> float:
    """Convert sweeper value in assembly units."""
    if kind is Parameter.amplitude:
        return value * MAX_PARAM[kind]
    if kind is Parameter.relative_phase:
        return (value / (2 * np.pi)) % 1.0 * MAX_PARAM[kind]
    if kind is Parameter.frequency:
        return value / 500e6 * MAX_PARAM[kind]
    if kind is Parameter.offset:
        return value * MAX_PARAM[kind]
    if kind is Parameter.duration:
        return value
    raise ValueError(f"Unsupported sweeper: {kind.name}")


def convert_or_pulse_duration(
    value: float, kind: Parameter, pulse: Optional[PulseLike], duration: int
) -> int:
    """Wrap :func:`convert` handling pulse duration sweep.

    In the special case of the duration sweep over an actual pulse, it
    is not possible to convert from the values in the original sweeper,
    since it needs to be first processed in order to generate the many
    waveforms corresponding to the shape evaluated for the desired
    amount of samples.
    In this case, the value is explicitly required, and it is set in the wrapper
    functions :func:`start` and :func:`step`.
    """
    return (
        duration
        if kind is Parameter.duration and isinstance(pulse, Pulse)
        else int(_convert(value, kind))
    )


def start(value: float, kind: Parameter, pulse: Optional[PulseLike]) -> int:
    """Convert sweeper start value in assembly units."""
    return convert_or_pulse_duration(value, kind, pulse, 0)


def step(value: float, kind: Parameter, pulse: Optional[PulseLike]) -> int:
    """Convert sweeper start value in assembly units."""
    return convert_or_pulse_duration(value, kind, pulse, 2)


def params(sweepers: list[ParallelSweepers], allocated: int) -> Params:
    """Initialize parameters' registers.

    `allocated` is the number of already allocated registers for loop counters, as
    initialized by :func:`loops`.
    """
    return [
        (
            j,
            Param(
                reg=Register(number=i + allocated + 1),
                start=start,
                step=step,
                pulse=pulse,
                channel=channel,
                kind=kind,
                sweeper=j,
            ),
        )
        for i, (j, start, step, pulse, channel, kind) in enumerate(
            (
                j,
                start(sweep.irange[0], sweep.parameter, pulse),
                step(sweep.irange[2], sweep.parameter, pulse),
                pulse.id if pulse is not None else None,
                channel,
                sweep.parameter,
            )
            for j, parsweep in enumerate(sweepers)
            for sweep in parsweep
            for pulse in (sweep.pulses if sweep.pulses is not None else [None])
            for channel in (sweep.channels if sweep.channels is not None else [None])
        )
    ]


class Update(Model):
    update: Optional[Callable[[Value], Instruction]]
    reset: Optional[Callable[[Value], Instruction]]


SWEEP_UPDATE: dict[Parameter, Update] = {
    Parameter.frequency: Update(update=lambda v: SetFreq(value=v), reset=None),
    Parameter.offset: Update(
        update=lambda v: SetAwgOffs(value_0=v, value_1=v), reset=None
    ),
    Parameter.amplitude: Update(
        update=lambda v: SetAwgGain(value_0=v, value_1=v),
        reset=lambda _: SetAwgGain(
            value_0=MAX_PARAM[Parameter.amplitude],
            value_1=MAX_PARAM[Parameter.amplitude],
        ),
    ),
    Parameter.relative_phase: Update(update=lambda v: SetPhDelta(value=v), reset=None),
    Parameter.duration: Update(update=None, reset=None),
}


def update_instructions(
    kind: Parameter, value: Value, reset: bool = False
) -> list[Instruction]:
    wrapper = SWEEP_UPDATE[kind]
    up = wrapper.update if not reset else wrapper.reset
    return [up(value)] if up is not None else []


def reset_instructions(kind: Parameter, value: Value) -> list[Instruction]:
    return update_instructions(kind, value, reset=True)


def _channels_pulses(
    pars: Iterable[tuple[int, Param]],
) -> tuple[list[Param], list[Param]]:
    channels = []
    pulses = []
    for p in pars:
        (channels if p[1].channel is not None else pulses).append(p[1])
    return channels, pulses


def params_reshape(params: Params) -> IndexedParams:
    """Split parameters related to channels and pulses.

    Moreover, it reorganize them by loop, to group the updates.
    """
    return {
        key: _channels_pulses(pars) for key, pars in groupby(params, key=lambda t: t[0])
    }


ParameterizedPulse = tuple[PulseLike, Optional[Param]]
SweepSequence = list[ParameterizedPulse]


def sweep_sequence(sequence: Iterable[PulseLike], params: list[Param]) -> SweepSequence:
    """Wrap swept pulses with updates markers."""
    parbyid = {p.pulse: p for p in params}
    return [(p, parbyid.get(p.id)) for p in sequence]
