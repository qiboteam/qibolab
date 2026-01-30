from collections.abc import Iterable
from typing import Annotated, Optional, Union

import numpy as np
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
) -> tuple[dict[WaveformInd, WaveformSpec], WaveformIndices]:
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

    cids = list(indexless.keys())
    bytes_ = [spec.waveform.data.tobytes() for spec in indexless.values()]
    durations = [spec.duration for spec in indexless.values()]
    waveforms = [spec.waveform for spec in indexless.values()]

    _, unique_idx, inverse_idx = np.unique(
        bytes_, return_index=True, return_inverse=True
    )

    waveform_indices = {
        cid: (inverse_idx[i], durations[i]) for i, cid in enumerate(cids)
    }

    waveform_specs = {
        idx: waveforms[np.where(inverse_idx == idx)[0][0]] for idx in unique_idx
    }

    return waveform_specs, waveform_indices
