from collections.abc import Iterable
from typing import Annotated, Union

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
    def _make_waveform(
        pulse: Pulse, component: str, duration: float | None = None, index: int = 0
    ) -> WaveformSpec:
        duration_ = pulse.duration if duration is None else duration
        update = (
            {"duration": duration_} | {"amplitude": 1.0}
            if pulse.id in amplitude_swept
            else {}
        )
        return WaveformSpec(
            waveform=Waveform(
                data=getattr(pulse.model_copy(update=update), component)(sampling_rate),
                index=index,
            ),
            duration=int(duration_),
        )

    def _sweep_durations(sweep: Sweeper):
        start, _, step = sweep.irange
        return [start + step * i for i in range(len(sweep))]

    pulses = [_pulse(e) for e in sequence if isinstance(e, (Pulse, Readout))]

    pulses_not_swept = [p for p in pulses if p not in duration_swept]
    pulses_swept = [
        (_pulse(p), duration_swept[p])
        for p in duration_swept
        if isinstance(p, (Pulse, Readout))
    ]

    # deduplicate non-swept pulses
    # NOTE: the reason swept pulses are not deduplicated is that they are swept over
    # duration so there will be no duplicates. It is still possible that a swept pulse
    # is the same as a non-swept pulse but this is not a prominent enough use-case to
    # justify accounting for it.
    hashes = np.array([hash(p) for p in pulses_not_swept])
    _, unique_idx, inverse_idx = np.unique(
        hashes, return_index=True, return_inverse=True
    )
    unique_pulses = [pulses_not_swept[i] for i in unique_idx]

    # the ids for the swept pulses start counting from `base` since up to here we need
    # two indices (i and q) for each unique non-swept pulse
    base = 2 * len(unique_pulses)

    # mapping from integer to unique WaveformSpec
    waveform_specs: dict[int, WaveformSpec] = {  # non-swept
        i * 2 + ch: _make_waveform(pulse, comp, index=i * 2 + ch)
        for i, pulse in enumerate(unique_pulses)
        for ch, comp in enumerate(("i", "q"))
    } | {  # swept
        base + 2 * k + ch: _make_waveform(
            pulse, comp, duration, index=base + 2 * k + ch
        )
        for k, (pulse, sweep) in enumerate(pulses_swept)
        for duration in _sweep_durations(sweep)
        for ch, comp in enumerate(("i", "q"))
    }

    # mapping that associate each element in the full list of pulses identified by
    # (UUID, i or q) to an integer that can be associated with a WaveformSpec through
    # waveform_specs
    idices_map: WaveformIndices = {  # non-swept
        (pulse.id, ch): (inv * 2 + ch, int(pulse.duration))
        for inv, pulse in zip(inverse_idx, pulses_not_swept)
        for ch, _ in enumerate(("i", "q"))
    } | {  # swept
        (pulse.id, 2 * i + ch): (base + 2 * k + ch, int(duration))
        for k, (pulse, sweep) in enumerate(pulses_swept)
        for i, duration in enumerate(_sweep_durations(sweep))
        for ch, _ in enumerate(("i", "q"))
    }

    return waveform_specs, idices_map
