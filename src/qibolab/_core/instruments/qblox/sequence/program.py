from enum import Enum

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
from qibolab._core.sweeper import ParallelSweepers, Parameter

from ..ast_ import (
    Acquire,
    Instruction,
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
from .waveforms import WaveformIndices

ComponentId = tuple[str, int]


class Registers(Enum):
    bin = Register(number=0)
    shots = Register(number=1)


def pulse_uid(pulse: Pulse) -> str:
    return str(hash(pulse))


def initialization(nshots: int) -> list:
    """Initialize registers."""
    return [Line(instruction=Move(source=nshots, destination=Register(number=0)))]


def setup() -> list:
    """Set up."""
    return [Line(instruction=WaitSync(duration=4))]


def execution(
    sequence: PulseSequence,
    waveforms: WaveformIndices,
    acquisitions: Acquisitions,
    sampling_rate: float,
) -> list[Line]:
    """The representation of the actual experiment to be executed."""
    return [
        Line.instr(i_)
        for block in (play(pulse, waveforms, sampling_rate) for _, pulse in sequence)
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


def loop(
    sweepers: list, experiment: list[Line], relaxation_time: int, outer_shots: bool
):
    shots = (Registers.shots.value, [Nop()])
    sweep = [
        (i, parameters_update(parsweep)) for i, parsweep in enumerate(sweepers, start=2)
    ]
    loops = [shots] + sweep if outer_shots else sweep + [shots]

    return (
        [
            inst
            for lp in loops
            for inst in (
                [Line(instruction=lp[1][0], label=f"l{lp[0]}")]
                + [Line(instruction=inst_) for inst_ in lp[1][1:]]
            )
        ]
        + experiment
        + [Line(instruction=Wait(duration=relaxation_time))]
        + [
            Line(
                instruction=Loop(
                    a=Register(number=lp[0]), address=Reference(label=f"l{lp[0]}")
                )
            )
            for lp in loops
        ]
    )


def finalization() -> list:
    """Finalize."""
    return [Line(instruction=Stop())]


PHASE_FACTOR = 1e9 / (2 * np.pi)


def play(
    pulse: PulseLike,
    waveforms: WaveformIndices,
    acquisitions: Acquisitions,
    sampling_rate: float,
) -> list[Instruction]:
    """Process the individual pulse in experiment."""
    if isinstance(pulse, Align):
        raise NotImplementedError("Align operation not yet supported by Qblox driver.")
    if isinstance(pulse, Readout):
        raise NotImplementedError(
            "Readout unsupported for Qblox - the operation should be unpacked in Pulse and Acquisition"
        )

    def _play(pulse: Pulse) -> Play:
        uid = pulse_uid(pulse)
        return Play(wave_0=waveforms[(uid, 0)], wave_1=waveforms[(uid, 1)], duration=0)

    return (
        [_play(pulse)]
        if isinstance(pulse, Pulse)
        else (
            [Wait(duration=int(pulse.duration * sampling_rate))]
            if isinstance(pulse, Delay)
            else (
                [SetPhDelta(value=int(pulse.phase * PHASE_FACTOR))]
                if isinstance(pulse, VirtualZ)
                else (
                    [
                        Acquire(
                            acquisition=acquisitions[str(pulse.id)].index,
                            bin=Registers.bin.value,
                            duration=0,
                        )
                    ]
                    if isinstance(pulse, Acquisition)
                    else []
                )
            )
        )
    )


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

    return Program(
        elements=initialization(options.nshots)
        + setup()
        + loop(
            sweepers,
            execution(sequence, waveforms, acquisitions, sampling_rate),
            relaxation_time=options.relaxation_time,
            outer_shots=options.averaging_mode is not AveragingMode.SEQUENTIAL,
        )
        + finalization()
    )
