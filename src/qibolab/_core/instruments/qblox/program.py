from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.pulses.pulse import Pulse, PulseLike
from qibolab._core.sequence import PulseSequence

from .ast_ import (
    Line,
    Loop,
    Move,
    Nop,
    Play,
    Program,
    Reference,
    Register,
    Stop,
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


def execution(sequence: PulseSequence, waveforms: WaveformIndices) -> list[Line]:
    """The representation of the actual experiment to be executed."""
    return [
        el
        for block in (play(pulse, waveforms) for _, pulse in sequence)
        for el in block
    ]


def loop(sweepers, experiment: list[Line]):
    return (
        [Line(instruction=Nop(), label="shots")]
        + experiment
        + [
            Line(
                instruction=Loop(a=Register(number=0), address=Reference(label="shots"))
            )
        ]
    )


def finalization() -> list:
    """Finalize."""
    return [Line(instruction=Stop())]


def play(pulse: PulseLike, waveforms: WaveformIndices) -> list[Line]:
    """Process the individual pulse in experiment."""
    return (
        [
            Line(
                instruction=Play(
                    wave_0=waveforms[(pulse_uid(pulse), 0)],
                    wave_1=waveforms[(pulse_uid(pulse), 1)],
                    duration=0,
                )
            )
        ]
        if isinstance(pulse, Pulse)
        else []
    )


def program(
    sequence: PulseSequence, waveforms: WaveformIndices, options: ExecutionParameters
):
    return Program(
        elements=initialization(options.nshots)
        + setup()
        + loop([], execution(sequence, waveforms))
        + finalization()
    )
