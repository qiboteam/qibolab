from collections.abc import Sequence
from enum import Enum
from typing import Optional, Union

import numpy as np

from qibolab._core.execution_parameters import AveragingMode, ExecutionParameters
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
from qibolab._core.sweeper import ParallelSweepers, Parameter, iteration_length

from ..q1asm.ast_ import (
    Acquire,
    Add,
    Instruction,
    Jge,
    Line,
    Loop,
    Move,
    Nop,
    Play,
    Program,
    Reference,
    Register,
    ResetPh,
    SetAwgGain,
    SetAwgOffs,
    SetFreq,
    SetPhDelta,
    Stop,
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


Param = tuple[Register, int, int, Optional[PulseId], str]

Params = Sequence[tuple[int, Param]]
"""Sequence of update parameters tuples.

These are produced by the :func:`params` function, and consist of a:

- :class:`Register`, used for the parameter value
- the initial value
- the increment
- the loop to which is associated
- the :class:`PulseId` of the target pulse (if the sweeper targets pulses)
- a textual description, used in some accompanying comments
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
    return 0 if kind is Parameter.duration else convert(value, kind)


def params(sweepers: list[ParallelSweepers], allocated: int) -> Params:
    """Initialize parameters' registers.

    `allocated` is the number of already allocated registers for loop counters, as
    initialized by :func:`loops`.
    """
    return [
        (
            j,
            (
                Register(number=i + allocated + 1),
                start,
                step,
                pulse,
                f"sweeper {j + 1} (pulse: {pulse})",
            ),
        )
        for i, (j, start, step, pulse) in enumerate(
            (
                j,
                start(sweep.irange[0], sweep.parameter),
                step(sweep.irange[2], sweep.parameter),
                pulse.id if pulse is not None else None,
            )
            for j, parsweep in enumerate(sweepers)
            for sweep in parsweep
            for pulse in (sweep.pulses if sweep.pulses is not None else [None])
        )
    ]


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
                instruction=Move(source=p[1], destination=p[0]),
                comment=f"init {p[4]}",
            )
            for p in params
        ]
        + [WaitSync(duration=4)]
    )


SWEEPERS = {
    Parameter.frequency: lambda v, o: ([SetFreq(value=v)], [SetFreq(value=o)]),
    Parameter.amplitude: lambda v, o: (
        [SetAwgGain(value_0=v, value_1=v)],
        [SetAwgGain(value_0=o, value_1=o)],
    ),
    Parameter.relative_phase: lambda v, o: (
        [SetPhDelta(value=v)],
        [SetPhDelta(value=-v)],
    ),
    Parameter.offset: lambda v, o: (
        [SetAwgOffs(value_0=v, value_1=v)],
        [SetAwgOffs(value_0=o, value_1=o)],
    ),
}


def parameters_update(sweepers: ParallelSweepers) -> list[Instruction]:
    return [Nop()]


SweepSequence = list[PulseLike]


def sweep_sequence(sequence: PulseSequence) -> SweepSequence:
    """Wrap swept pulses with updates markers."""
    return [p for _, p in sequence]


def execution(
    sequence: SweepSequence,
    waveforms: WaveformIndices,
    acquisitions: Acquisitions,
    sampling_rate: float,
) -> list[Instruction]:
    """The representation of the actual experiment to be executed."""
    return [
        i_
        for block in (
            play(pulse, waveforms, acquisitions, sampling_rate) for pulse in sequence
        )
        for i_ in block
    ]


START = "start"
SHOTS = "shots"


def loop_conclusion(relaxation_time: int):
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


def loop_machinery(loops: Loops, singleshot: bool):
    def shots(marker: Optional[str]):
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
            Line(
                instruction=Loop(a=lp[0], address=Reference(label=START)),
                comment="loop over " + ("shots" if lp[2] is None else lp[2]),
                label=SHOTS if shots(lp[2]) else None,
            ),
            Move(source=lp[1], destination=lp[0]),
        ]
    ][:-1]


def loop(
    loops: Loops, experiment: list[Instruction], relaxation_time: int, singleshot: bool
) -> Sequence[Union[Line, Instruction]]:
    conclusion = loop_conclusion(relaxation_time)
    machinery = loop_machinery(loops, singleshot)
    main = experiment + conclusion + machinery

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


def play(
    pulse: PulseLike,
    waveforms: WaveformIndices,
    acquisitions: Acquisitions,
    sampling_rate: float,
) -> list[Instruction]:
    """Process the individual pulse in experiment."""

    if isinstance(pulse, Pulse):
        uid = pulse_uid(pulse)
        return [
            Play(wave_0=waveforms[(uid, 0)], wave_1=waveforms[(uid, 1)], duration=0)
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


def program(
    sequence: PulseSequence,
    waveforms: WaveformIndices,
    acquisitions: Acquisitions,
    options: ExecutionParameters,
    sweepers: list[ParallelSweepers],
    sampling_rate: float,
) -> Program:
    assert options.nshots is not None
    assert options.relaxation_time is not None

    loops_ = loops(
        sweepers,
        options.nshots,
        inner_shots=options.averaging_mode is AveragingMode.SEQUENTIAL,
    )
    params_ = params(sweepers, start=max(lp[0].number for lp in loops_))
    sweepseq = sweep_sequence(sequence, params_)

    return Program(
        elements=[
            el if isinstance(el, Line) else Line.instr(el)
            for block in [
                setup(loops_, params_),
                loop(
                    loops_,
                    execution(sweepseq, waveforms, acquisitions, sampling_rate),
                    options.relaxation_time,
                    options.averaging_mode is AveragingMode.SINGLESHOT,
                ),
                finalization(),
            ]
            for el in block
        ]
    )
