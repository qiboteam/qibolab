from collections.abc import Iterable, Sequence
from enum import Enum
from itertools import groupby
from typing import Callable, Optional, Union, cast

import numpy as np

from qibolab._core.execution_parameters import AveragingMode, ExecutionParameters
from qibolab._core.identifier import ChannelId
from qibolab._core.instruments.qblox.q1asm.ast_ import SetAwgGain, SetAwgOffs, SetFreq
from qibolab._core.pulses.pulse import (
    Acquisition,
    Align,
    Delay,
    Pulse,
    PulseId,
    PulseLike,
    Readout,
    VirtualZ,
)
from qibolab._core.sequence import PulseSequence
from qibolab._core.serialize import Model
from qibolab._core.sweeper import ParallelSweepers, Parameter, iteration_length

from ..q1asm.ast_ import (
    Acquire,
    Add,
    Instruction,
    Jge,
    Line,
    Loop,
    Move,
    Play,
    Program,
    Reference,
    Register,
    ResetPh,
    SetPhDelta,
    Stop,
    Value,
    Wait,
    WaitSync,
)
from .acquisition import Acquisitions
from .waveforms import WaveformIndices, pulse_uid

__all__ = ["Program"]


class Registers(Enum):
    bin = Register(number=0)
    bin_reset = Register(number=1)
    shots = Register(number=2)


Loops = Sequence[tuple[Register, int, Optional[int]]]
"""Sequence of loop-characterizing tuples.

These are produced by the :func:`loops` function, and consist of a:

- :class:`Register`, which is used as a loop counter, or it is auxiliary to the
  iteration process
- the iteration length
- the iteration index
"""


def loops(sweepers: list[ParallelSweepers], nshots: int, inner_shots: bool) -> Loops:
    """Initialize registers for loop counters.

    The counters implement the ``length`` of the iteration, which, for a general
    sweeper, is fully characterized by a ``(start, step, length)`` tuple.

    Those related to :attr:`Registers.bin` and :attr:`Registers.bin_reset` are actually
    not loop counter on their own, but they are required to properly store the
    acquisitions in different bins.
    """
    shots = (Registers.shots.value, nshots, None)
    first_sweeper = max(r.value.number for r in Registers) + 1
    sweep = [
        (
            Register(number=i + first_sweeper),
            iteration_length(parsweep),
            i,
        )
        for i, parsweep in enumerate(sweepers)
    ]
    return [shots] + sweep if inner_shots else sweep + [shots]


def sweep_desc(index: int) -> str:
    """Sweeper textual description."""
    return f"sweeper {index + 1}"


class Param(Model):
    register: Register
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

MAX_PARAM = {
    Parameter.amplitude: 32767,
    Parameter.offset: 32767,
    Parameter.relative_phase: 1e9,
    Parameter.frequency: 2e9,
}
"""Maximum range for parameters.

Declared in https://docs.qblox.com/en/main/cluster/q1_sequence_processor.html#q1-instructions

Ranges may be one-sided (just positive) or two-sided. This is accounted for in
:func:`convert`.
"""


def convert(value: float, kind: Parameter) -> int:
    """Convert sweeper value in assembly units."""
    if kind is Parameter.amplitude:
        return int(value * MAX_PARAM[kind])
    if kind is Parameter.relative_phase:
        return int((value / (2 * np.pi)) % 1.0 * MAX_PARAM[kind])
    if kind is Parameter.frequency:
        return int(value / 5e8 * MAX_PARAM[kind])
    if kind is Parameter.offset:
        return int(value * MAX_PARAM[kind])
    raise ValueError(f"Unsupported sweeper: {kind.name}")


def start(value: float, kind: Parameter) -> int:
    """Convert sweeper start value in assembly units."""
    return 0 if kind is Parameter.duration else convert(value, kind)


def step(value: float, kind: Parameter) -> int:
    """Convert sweeper start value in assembly units."""
    return 2 if kind is Parameter.duration else convert(value, kind)


def params(sweepers: list[ParallelSweepers], allocated: int) -> Params:
    """Initialize parameters' registers.

    `allocated` is the number of already allocated registers for loop counters, as
    initialized by :func:`loops`.
    """
    return [
        (
            j,
            Param(
                register=Register(number=i + allocated + 1),
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
                start(sweep.irange[0], sweep.parameter),
                step(sweep.irange[2], sweep.parameter),
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


def setup(loops: Loops, params: list[Param]) -> Sequence[Union[Line, Instruction]]:
    """Set up."""
    return (
        [
            Line(
                instruction=Move(source=0, destination=Registers.bin.value),
                comment="init bin counter",
            ),
            Line(
                instruction=Move(source=0, destination=Registers.bin_reset.value),
                comment="init bin reset",
            ),
        ]
        + [
            Line(
                instruction=Move(source=lp[1], destination=lp[0]),
                comment="init "
                + ("shots" if lp[2] is None else sweep_desc(lp[2]))
                + " counter",
            )
            for lp in loops
        ]
        + [
            Line(
                instruction=Move(source=p.start, destination=p.register),
                comment=f"init {p.description}",
            )
            for p in params
        ]
        + [
            inst
            for p in params
            if p.channel is not None
            for inst in update_instructions(p.kind, p.start)
        ]  # TODO: condition on the ID of the current channel, since the sequence is being built for it
        + [WaitSync(duration=4)]
    )


IndexedParams = dict[int, tuple[list[Param], list[Param]]]


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


def sweep_sequence(sequence: PulseSequence, params: list[Param]) -> SweepSequence:
    """Wrap swept pulses with updates markers."""
    parbyid = {p.pulse: p for p in params}
    return [(p, parbyid.get(p.id)) for _, p in sequence]


def execution(
    sequence: SweepSequence,
    waveforms: WaveformIndices,
    acquisitions: Acquisitions,
    sampling_rate: float,
) -> list[Instruction]:
    """Representation of the actual experiment to be executed."""
    return [
        i_
        for block in (
            event(pulse, waveforms, acquisitions, sampling_rate) for pulse in sequence
        )
        for i_ in block
    ]


START = "start"
SHOTS = "shots"


def iteration_end(relaxation_time: int) -> Sequence[Line]:
    return [
        Line(instruction=Wait(duration=relaxation_time), comment="relaxation"),
        Line(instruction=ResetPh(), comment="phase reset"),
        Line(
            instruction=Add(
                a=Registers.bin.value, b=1, destination=Registers.bin.value
            ),
            comment="bin increment",
        ),
    ]


def loop_machinery(
    loops: Loops, params: IndexedParams, singleshot: bool, channel: ChannelId
) -> Sequence[Union[Line, Instruction]]:
    def shots(marker: Optional[int]) -> bool:
        return marker is None and not singleshot

    return [
        i_
        for lp in loops
        for i_ in (
            [
                Line(
                    instruction=Jge(
                        a=Registers.shots.value, b=1, address=Reference(label=SHOTS)
                    ),
                    comment="skip bin reset - advance both counters",
                ),
                Line(
                    instruction=Move(
                        source=Registers.bin_reset.value,
                        destination=Registers.bin.value,
                    ),
                    comment="shots average: reset bin counter",
                ),
            ]
            if shots(lp[2])
            else []
        )
        + [
            *(
                (
                    line
                    for block in (
                        (
                            Line(
                                instruction=Add(
                                    a=p.register, b=p.step, destination=p.register
                                ),
                                comment=f"increment {p.description}",
                            ),
                            *(
                                update_instructions(p.kind, p.register)
                                if p.description is not None and p.channel == channel
                                else ()
                            ),
                        )
                        for p in (params[lp[2]][0] + params[lp[2]][1])
                    )
                    for line in block
                )
                if lp[2] is not None
                else ()
            ),
            Line(
                instruction=Loop(a=lp[0], address=Reference(label=START)),
                comment="loop over "
                + ("shots" if lp[2] is None else sweep_desc(lp[2])),
                label=SHOTS if shots(lp[2]) else None,
            ),
            Move(source=lp[1], destination=lp[0]),
        ]
    ][:-1]


def loop(
    loops: Loops,
    params: IndexedParams,
    experiment: list[Instruction],
    relaxation_time: int,
    singleshot: bool,
    channel: ChannelId,
) -> Sequence[Union[Line, Instruction]]:
    end = cast(list, iteration_end(relaxation_time))
    machinery = cast(list, loop_machinery(loops, params, singleshot, channel))
    main = experiment + end + machinery

    return [
        (
            Line(instruction=main[0], label=START)
            if isinstance(main[0], Instruction)
            else Line(
                instruction=main[0].instruction, comment=main[0].comment, label=START
            )
        )
    ] + main[1:]


def finalization() -> list[Instruction]:
    """Finalize."""
    return [Stop()]


PHASE_FACTOR = 1e9 / (2 * np.pi)


def play_pulse(pulse: Pulse, waveforms: WaveformIndices) -> Instruction:
    uid = pulse_uid(pulse)
    return Play(wave_0=waveforms[(uid, 0)], wave_1=waveforms[(uid, 1)], duration=0)


def play_duration_swept(pulse: Pulse, param: Param) -> Instruction:
    return Play(
        wave_0=param.register,
        wave_1=Register(number=param.register.number + 1),
        duration=0,
    )


def play(
    parpulse: ParameterizedPulse,
    waveforms: WaveformIndices,
    acquisitions: Acquisitions,
    sampling_rate: float,
) -> list[Instruction]:
    """Process the individual pulse in experiment."""
    pulse = parpulse[0]
    param = parpulse[1]

    if isinstance(pulse, Pulse):
        return [
            play_pulse(pulse, waveforms)
            if param is None or param.kind is not Parameter.duration
            else play_duration_swept(pulse, param)
        ]
    if isinstance(pulse, Delay):
        return [Wait(duration=int(pulse.duration * sampling_rate))]
    if isinstance(pulse, VirtualZ):
        return [SetPhDelta(value=int(pulse.phase * PHASE_FACTOR))]
    if isinstance(pulse, Acquisition):
        return [
            Acquire(
                acquisition=acquisitions[str(pulse.id)].index,
                bin=Registers.bin.value,
                duration=0,
            )
        ]
    if isinstance(pulse, Align):
        raise NotImplementedError("Align operation not yet supported by Qblox.")
    if isinstance(pulse, Readout):
        raise NotImplementedError(
            "Readout unsupported for Qblox - the operation should be unpacked in Pulse and Acquisition"
        )
    raise NotImplementedError(f"Instruction {type(pulse)} unsupported by Qblox driver.")


def event(
    parpulse: ParameterizedPulse,
    waveforms: WaveformIndices,
    acquisitions: Acquisitions,
    sampling_rate: float,
) -> list[Instruction]:
    param = parpulse[1]
    return (
        (update_instructions(param.kind, param.register) if param is not None else [])
        + play(parpulse, waveforms, acquisitions, sampling_rate)
        + (reset_instructions(param.kind, param.register) if param is not None else [])
    )


def program(
    sequence: PulseSequence,
    waveforms: WaveformIndices,
    acquisitions: Acquisitions,
    options: ExecutionParameters,
    sweepers: list[ParallelSweepers],
    sampling_rate: float,
    channel: ChannelId,
) -> Program:
    assert options.nshots is not None
    assert options.relaxation_time is not None

    loops_ = loops(
        sweepers,
        options.nshots,
        inner_shots=options.averaging_mode is AveragingMode.SEQUENTIAL,
    )
    params_ = params(sweepers, allocated=max(lp[0].number for lp in loops_))
    indexed_params = params_reshape(params_)
    sweepseq = sweep_sequence(
        sequence, [p for v in indexed_params.values() for p in v[1]]
    )

    return Program(
        elements=[
            el if isinstance(el, Line) else Line.instr(el)
            for block in [
                setup(loops_, [p for _, p in params_]),
                loop(
                    loops_,
                    indexed_params,
                    execution(sweepseq, waveforms, acquisitions, sampling_rate),
                    options.relaxation_time,
                    options.averaging_mode is AveragingMode.SINGLESHOT,
                    channel,
                ),
                finalization(),
            ]
            for el in block
        ]
    )
