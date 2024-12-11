import numpy as np

from qibolab._core.execution_parameters import AveragingMode, ExecutionParameters
from qibolab._core.pulses.pulse import Delay, Pulse, PulseLike, VirtualZ
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers, Parameter

from .ast_ import (
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

ComponentId = tuple[str, int]
WaveformIndices = dict[ComponentId, int]


def pulse_uid(pulse: Pulse) -> str:
    return str(hash(pulse))


def initialization(nshots: int) -> list:
    """Initialize registers."""
    return [Line(instruction=Move(source=nshots, destination=Register(number=0)))]


def setup() -> list:
    """Set up."""
    return [Line(instruction=WaitSync(duration=4))]


def execution(
    sequence: PulseSequence, waveforms: WaveformIndices, sampling_rate: float
) -> list[Line]:
    """The representation of the actual experiment to be executed."""
    return [
        el
        for block in (play(pulse, waveforms, sampling_rate) for _, pulse in sequence)
        for el in block
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
    shots = (0, [Nop()])
    sweep = [
        (i, parameters_update(parsweep)) for i, parsweep in enumerate(sweepers, start=1)
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
    pulse: PulseLike, waveforms: WaveformIndices, sampling_rate: float
) -> list[Line]:
    """Process the individual pulse in experiment."""
    return (
        [
            Line.instr(
                Play(
                    wave_0=waveforms[(pulse_uid(pulse), 0)],
                    wave_1=waveforms[(pulse_uid(pulse), 1)],
                    duration=0,
                )
            )
        ]
        if isinstance(pulse, Pulse)
        else (
            [Line.instr(Wait(duration=int(pulse.duration * sampling_rate)))]
            if isinstance(pulse, Delay)
            else (
                [Line.instr(SetPhDelta(value=int(pulse.phase * PHASE_FACTOR)))]
                if isinstance(pulse, VirtualZ)
                else []
            )
        )
    )


def program(
    sequence: PulseSequence,
    waveforms: WaveformIndices,
    options: ExecutionParameters,
    sweepers: list[ParallelSweepers],
    sampling_rate: float,
):
    return Program(
        elements=initialization(options.nshots)
        + setup()
        + loop(
            sweepers,
            execution(sequence, waveforms, sampling_rate),
            relaxation_time=options.relaxation_time,
            outer_shots=options.averaging_mode is not AveragingMode.SEQUENTIAL,
        )
        + finalization()
    )
