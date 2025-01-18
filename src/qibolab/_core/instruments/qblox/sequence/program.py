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
    PulseLike,
    Readout,
    VirtualZ,
)
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers, Parameter, iteration_length

from ..ast_ import (
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


class Registers(Enum):
    bin = Register(number=0)
    bin_reset = Register(number=1)
    shots = Register(number=2)


Loops = Sequence[tuple[Register, int, Optional[str]]]


def loops(sweepers: list[ParallelSweepers], nshots: int, inner_shots: bool) -> Loops:
    shots = (Registers.shots.value, nshots, None)
    first_sweeper = max(r.value.number for r in Registers) + 1
    sweep = [
        (
            Register(number=i + first_sweeper),
            iteration_length(parsweep),
            f"{i + 1}th sweeper",
        )
        for i, parsweep in enumerate(sweepers)
    ]
    return [shots] + sweep if inner_shots else sweep + [shots]


def setup(loops: Loops) -> Sequence[Union[Line, Instruction]]:
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
                comment="init " + ("shots" if lp[2] is None else lp[2]) + " counter",
            )
            for lp in loops
        ]
        + [WaitSync(duration=4)]
    )


def execution(
    sequence: PulseSequence,
    waveforms: WaveformIndices,
    acquisitions: Acquisitions,
    sampling_rate: float,
) -> list[Instruction]:
    """The representation of the actual experiment to be executed."""
    return [
        i_
        for block in (
            play(pulse, waveforms, acquisitions, sampling_rate) for _, pulse in sequence
        )
        for i_ in block
    ]


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


START = "start"
SHOTS = "shots"


def loop_conclusion(relaxation_time: int):
    return [
        Line(instruction=Wait(duration=relaxation_time), comment="relaxation"),
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
):
    assert options.nshots is not None
    assert options.relaxation_time is not None

    loops_ = loops(
        sweepers,
        options.nshots,
        inner_shots=options.averaging_mode is AveragingMode.SEQUENTIAL,
    )

    return Program(
        elements=[
            el if isinstance(el, Line) else Line.instr(el)
            for block in [
                setup(loops_),
                loop(
                    loops_,
                    execution(sequence, waveforms, acquisitions, sampling_rate),
                    options.relaxation_time,
                    options.averaging_mode is AveragingMode.SINGLESHOT,
                ),
                finalization(),
            ]
            for el in block
        ]
    )
