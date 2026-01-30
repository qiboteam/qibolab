from collections.abc import Iterable
from typing import Annotated, Optional, Union

import numpy as np
from pydantic import UUID4, AfterValidator

from qibolab._core.pulses import Pulse, PulseId, PulseLike, Readout
from qibolab._core.serialize import ArrayList, Model
from qibolab._core.sweeper import Sweeper

__all__ = []

ComponentId = tuple[UUID4, int]
WaveformIndex = int
WaveformIndices = dict[ComponentId, tuple[WaveformIndex, int]]


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
) -> tuple[dict[WaveformIndex, WaveformSpec], WaveformIndices]:
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

    pulses_not_swept = np.array(
        [
            pulse
            for pulse in (
                _pulse(event)
                for event in sequence
                if isinstance(event, (Pulse, Readout))
            )
            if pulse not in duration_swept
        ]
    )
    hashes_pulse_not_swept = [hash(pulse) for pulse in pulses_not_swept]
    _unique_hashes, unique_indices_not_swept, inverse_indices_not_swept = np.unique(
        hashes_pulse_not_swept, return_index=True, return_inverse=True
    )

    pulses_swept = [
        (pulse, sweep)
        for pulse, sweep in (
            (_pulse(event), duration_swept[event])
            for event in duration_swept
            if isinstance(event, (Pulse, Readout))
        )
    ]

    indexless = {
        k: v
        for d in (
            {
                (pulse.id, 0): _waveform(pulse, "i"),
                (pulse.id, 1): _waveform(pulse, "q"),
            }
            for pulse in pulses_not_swept[unique_indices_not_swept]
        )
        for k, v in d.items()
    } | {
        k: v
        for d in (
            {
                (pulse.id, 2 * i): _waveform(pulse, "i", duration),
                (pulse.id, 2 * i + 1): _waveform(pulse, "q", duration),
            }
            for pulse, sweep in pulses_swept
            for i, duration in (
                (i, sweep.irange[0] + sweep.irange[2] * i) for i in range(len(sweep))
            )
        )
        for k, v in d.items()
    }

    indices_not_swept = {
        (pulse.id, ch): (int(i * 2 + ch), pulse.duration)
        for pulse, i in zip(pulses_not_swept, inverse_indices_not_swept)
        for ch in (0, 1)
    }

    specs_not_swept = {
        int(inv * 2 + ch): WaveformSpec(
            waveform=Waveform(
                data=indexless[(pulse.id, ch)].waveform.data, index=int(inv * 2 + ch)
            ),
            duration=indexless[(pulse.id, ch)].duration,
        )
        for pulse, inv in zip(
            pulses_not_swept[unique_indices_not_swept], unique_indices_not_swept
        )
        for ch in (0, 1)
    }

    base = 2 * len(unique_indices_not_swept)

    indices_swept = {
        (pulse.id, ch + 2 * i): (base + 2 * k + ch, int(duration))
        for k, (pulse, sweep) in enumerate(pulses_swept)
        for i, duration in (
            (i, sweep.irange[0] + sweep.irange[2] * i) for i in range(len(sweep))
        )
        for ch in (0, 1)
    }

    specs_swept = {
        base + 2 * k + ch: WaveformSpec(
            waveform=Waveform(
                data=indexless[(pulse.id, 2 * i + ch)].waveform.data,
                index=base + 2 * k + ch,
            ),
            duration=int(duration),
        )
        for k, (pulse, sweep) in enumerate(pulses_swept)
        for i, duration in (
            (i, sweep.irange[0] + sweep.irange[2] * i) for i in range(len(sweep))
        )
        for ch in (0, 1)
    }

    waveform_indices = indices_not_swept | indices_swept
    waveform_specs = specs_not_swept | specs_swept

    return waveform_specs, waveform_indices
