from qibolab._core.pulses.pulse import Pulse, PulseLike
from qibolab._core.sequence import PulseSequence

from .ast_ import Line, Play, Program

ComponentId = tuple[str, int]
WaveformIndices = dict[ComponentId, int]


def pulse_uid(pulse: Pulse) -> str:
    return str(hash(pulse))


def instructions(pulse: PulseLike, waveforms: WaveformIndices) -> list:
    breakpoint()
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


def program(sequence: PulseSequence, waveforms: WaveformIndices):
    return Program(
        elements=[
            el
            for block in (instructions(pulse, waveforms) for _, pulse in sequence)
            for el in block
        ]
    )
