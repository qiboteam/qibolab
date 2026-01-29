from collections.abc import Iterable
from typing import Annotated, Optional, Union

from pydantic import UUID4, AfterValidator

from qibolab._core.pulses import Pulse, PulseId, PulseLike, Readout
from qibolab._core.serialize import ArrayList, Model
from qibolab._core.sweeper import Sweeper

__all__ = []

ComponentId = tuple[UUID4, int]
WaveformInd = int
WaveformIndices = dict[ComponentId, tuple[WaveformInd, int]]


class Waveform(Model):
    data: Annotated[ArrayList, AfterValidator(lambda a: a.astype(float))]
    index: int


class WaveformSpec(Model):
    waveform: Waveform
    duration: int


def _pulse(event: Union[Pulse, Readout]) -> Pulse:
    return event.probe if isinstance(event, Readout) else event


def waveforms(
    sequence: Iterable[PulseLike],
    sampling_rate: float,
    amplitude_swept: set[PulseId],
    duration_swept: dict[PulseLike, Sweeper],
) -> Union[dict[WaveformInd, WaveformSpec], WaveformIndices]:
    def _waveform(
        pulse: Pulse, component: str, duration: Optional[float] = None
    ) -> WaveformSpec:
        duration_ = pulse.duration if duration is None else duration
        update = {"duration": duration_} | (
            {"amplitude": 1.0} if pulse.id in amplitude_swept else {}
        )
        return WaveformSpec(
            waveform=Waveform(
                data=getattr(pulse.model_copy(update=update), component)(sampling_rate),
                index=0,
            ),
            duration=int(duration_),
        )

    indexless = {
        k: v
        for d in (
            {
                (pulse.id, 0): _waveform(pulse, "i"),
                (pulse.id, 1): _waveform(pulse, "q"),
            }
            for pulse in (
                _pulse(event)
                for event in sequence
                if isinstance(event, (Pulse, Readout))
            )
            if pulse not in duration_swept
        )
        for k, v in d.items()
    } | {
        k: v
        for d in (
            {
                (pulse.id, 2 * i): _waveform(pulse, "i", duration),
                (pulse.id, 2 * i + 1): _waveform(pulse, "q", duration),
            }
            for pulse, sweep in (
                (_pulse(event), duration_swept[event])
                for event in duration_swept
                if isinstance(event, (Pulse, Readout))
            )
            for i, duration in (
                (i, sweep.irange[0] + sweep.irange[2] * i) for i in range(len(sweep))
            )
        )
        for k, v in d.items()
    }

    cid_to_key = {cid: spec.waveform.data.tobytes() for cid, spec in indexless.items()}
    unique_keys = set(cid_to_key.values())
    # key_to_index serves no puropse other than to make the output more readable.
    key_to_index = {k: i for i, k in enumerate(unique_keys)}

    unique_specs = {
        k: (
            indexless[next(cid for cid, kk in cid_to_key.items() if kk == k)],
            key_to_index[k],
        )
        for k in unique_keys
    }

    indices_map = {
        cid: (key_to_index[k], indexless[cid].duration) for cid, k in cid_to_key.items()
    }

    waveform_specs = {
        i: WaveformSpec(
            waveform=Waveform(data=spec.waveform.data, index=i), duration=spec.duration
        )
        for spec, i in unique_specs.values()
    }

    return waveform_specs, indices_map
