from collections.abc import Callable, Iterable
from enum import Enum, auto
from itertools import groupby
from typing import Optional

from qibolab._core.identifier import ChannelId
from qibolab._core.instruments.qblox.q1asm.ast_ import SetAwgGain, SetAwgOffs, SetFreq
from qibolab._core.pulses.pulse import (
    Pulse,
    PulseId,
    PulseLike,
)
from qibolab._core.serialize import Model
from qibolab._core.sweeper import ParallelSweepers, Parameter, Range, Sweeper

from ..q1asm.ast_ import (
    Instruction,
    Register,
    SetPhDelta,
    Value,
)
from .asm import MAX_PARAM, convert

__all__ = []


class ParamRole(Enum):
    """Parameter role in sweep.

    Discriminate the various register roles. Particularly, in the pulse
    duration sweeps.
    """

    FREQUENCY = auto(), Parameter.frequency
    "Channel frequency."
    OFFSET = auto(), Parameter.offset
    "Channel offset."
    AMPLITUDE = auto(), Parameter.amplitude
    "Pulse amplitude."
    PHASE = auto(), Parameter.relative_phase
    "Pulse relative phase."
    DURATION = auto(), Parameter.duration
    "Pulse duration."
    PULSE_I = auto(), Parameter.duration
    "Pulse I component."
    PULSE_Q = auto(), Parameter.duration
    "Pulse Q component."

    @classmethod
    def from_sweeper(cls, sweep: Sweeper) -> "ParamRole":
        for var in cls:
            if sweep.parameter == var.value[1]:
                return var
        raise ValueError("Sweeper parameter kind does not correspond to any role.")

    @classmethod
    def unique(cls, sweep: Sweeper) -> bool:
        return sweep.parameter is not Parameter.duration or (
            sweep.pulses is not None
            and not any(isinstance(p, Pulse) for p in sweep.pulses)
        )

    @property
    def kind(self):
        return self.value[1]


class Param(Model):
    reg: Register
    """Register used for the parameter value."""
    start: int
    """Initial value."""
    step: int
    """Increment."""
    role: ParamRole
    """The parameter type."""
    pulse: Optional[PulseId]
    """The target pulse (if the sweeper targets pulses)."""
    channel: Optional[ChannelId]
    """The target channel (if the sweeper targets channels)."""
    loop: Optional[int] = None
    """The loop which is associated to."""

    @property
    def description(self) -> str:
        """Textual description, used in some accompanying comments."""
        return (
            "sweeper ("
            + (f"loop: {self.loop}, " if self.loop is not None else "")
            + (f"pulse: 0x{self.pulse.hex[:5]}, " if self.pulse is not None else "")
            + f"role: {self.role.kind.name}.{self.role.name}"
            + ")"
        )


IndexedParams = dict[int, tuple[list[Param], list[Param]]]


def _pulse_duration(sweep: Sweeper) -> list[tuple[Range, "ParamRole"]]:
    """Reserve 3 registers for a pulse duration sweeper."""
    return [
        ((0, 2 * len(sweep), 2), ParamRole.PULSE_I),
        ((1, 2 * len(sweep) + 1, 2), ParamRole.PULSE_Q),
        ((sweep - 4.0).irange, ParamRole.DURATION),
    ]


def _registers(sweep: Sweeper) -> list[tuple[Range, ParamRole]]:
    """Reserve registers for sweeping."""
    return (
        [(sweep.irange, ParamRole.from_sweeper(sweep))]
        if ParamRole.unique(sweep)
        else _pulse_duration(sweep)
    )


def _unravel_sweeps(sweepers: list[ParallelSweepers]) -> Iterable[tuple[int, Param]]:
    """Turn sweepers into suitable ranges with unique targets."""
    return (
        (
            j,
            Param(
                reg=Register(number=0),
                start=int(convert(irange[0], sweep.parameter)),
                step=int(convert(irange[2], sweep.parameter)),
                pulse=pulse.id if pulse is not None else None,
                channel=channel,
                role=role,
            ),
        )
        for j, parsweep in enumerate(sweepers)
        for sweep in parsweep
        for irange, role in _registers(sweep)
        for pulse in (sweep.pulses if sweep.pulses is not None else [None])
        for channel in (sweep.channels if sweep.channels is not None else [None])
    )


def params(sweepers: list[ParallelSweepers], allocated: int) -> list[Param]:
    """Initialize parameters' registers.

    `allocated` is the number of already allocated registers for loop counters, as
    initialized by :func:`loops`.
    """
    return [
        p.model_copy(update={"reg": Register(number=i + allocated + 1), "loop": j})
        for i, (j, p) in enumerate(_unravel_sweeps(sweepers))
    ]


class _Update(Model):
    update: Optional[Callable[[Value], Instruction]]
    reset: Optional[Callable[[Value], Instruction]]


_SWEEP_UPDATE: dict[Parameter, _Update] = {
    Parameter.frequency: _Update(update=lambda v: SetFreq(value=v), reset=None),
    Parameter.offset: _Update(
        update=lambda v: SetAwgOffs(value_0=v, value_1=v), reset=None
    ),
    Parameter.amplitude: _Update(
        update=lambda v: SetAwgGain(value_0=v, value_1=v),
        reset=lambda _: SetAwgGain(
            value_0=MAX_PARAM[Parameter.amplitude],
            value_1=MAX_PARAM[Parameter.amplitude],
        ),
    ),
    Parameter.relative_phase: _Update(update=lambda v: SetPhDelta(value=v), reset=None),
    Parameter.duration: _Update(update=None, reset=None),
}


def update_instructions(
    role: ParamRole, value: Value, reset: bool = False
) -> list[Instruction]:
    wrapper = _SWEEP_UPDATE[role.kind]
    up = wrapper.update if not reset else wrapper.reset
    return [up(value)] if up is not None else []


def reset_instructions(role: ParamRole, value: Value) -> list[Instruction]:
    return update_instructions(role, value, reset=True)


def _channels_pulses(
    pars: Iterable[Param],
) -> tuple[list[Param], list[Param]]:
    channels = []
    pulses = []
    for p in pars:
        (channels if p.channel is not None else pulses).append(p)
    return channels, pulses


def params_reshape(params: list[Param]) -> IndexedParams:
    """Split parameters related to channels and pulses.

    Moreover, it reorganize them by loop, to group the updates.
    """
    return {
        key: _channels_pulses(pars)
        for key, pars in groupby(params, key=lambda p: p.loop)
        if key is not None
    }


ParameterizedPulse = tuple[PulseLike, set[Param]]
SweepSequence = list[ParameterizedPulse]


def sweep_sequence(sequence: Iterable[PulseLike], params: list[Param]) -> SweepSequence:
    """Wrap swept pulses with updates markers."""
    parsbyid = {
        p: {pair[1] for pair in pairs}
        for p, pairs in groupby(
            sorted(
                ((p.pulse, p) for p in params if p.pulse is not None),
                key=lambda pair: pair[0],
            ),
            key=lambda pair: pair[0],
        )
    }
    return [(e, parsbyid.get(e.id, set())) for e in sequence]
