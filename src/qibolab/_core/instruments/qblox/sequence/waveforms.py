from typing import Annotated, Union

from pydantic import AfterValidator

from qibolab._core.pulses import Pulse, Readout
from qibolab._core.sequence import PulseSequence
from qibolab._core.serialize import ArrayList, Model

__all__ = []

ComponentId = tuple[str, int]
WaveformIndices = dict[ComponentId, int]


def pulse_uid(pulse: Pulse) -> str:
    return str(hash(pulse))


class Waveform(Model):
    data: Annotated[ArrayList, AfterValidator(lambda a: a.astype(float))]
    index: int


Waveforms = dict[ComponentId, Waveform]


def waveforms(sequence: PulseSequence, sampling_rate: float) -> Waveforms:
    def waveform(pulse: Pulse, component: str) -> Waveform:
        return Waveform(data=getattr(pulse, component)(sampling_rate), index=0)

    def pulse_(event: Union[Pulse, Readout]) -> Pulse:
        return event.probe if isinstance(event, Readout) else event

    indexless = {
        k: v
        for d in (
            {
                (pulse_uid(pulse_(event)), 0): waveform(pulse_(event), "i"),
                (pulse_uid(pulse_(event)), 1): waveform(pulse_(event), "q"),
            }
            for _, event in sequence
            if isinstance(event, (Pulse, Readout))
        )
        for k, v in d.items()
    }

    return {
        k: Waveform(data=v.data, index=i) for i, (k, v) in enumerate(indexless.items())
    }


def waveform_indices(waveforms: Waveforms) -> WaveformIndices:
    return {k: w.index for k, w in waveforms.items()}
