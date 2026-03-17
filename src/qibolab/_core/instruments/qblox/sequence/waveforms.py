from collections.abc import Iterable, Sequence
from typing import Annotated, Union, cast

import numpy as np
import numpy.typing as npt
from pydantic import UUID4, AfterValidator

from qibolab._core.pulses import Pulse, PulseId, PulseLike, Readout
from qibolab._core.serialize import ArrayList, Model
from qibolab._core.sweeper import Sweeper

__all__ = []

QuadratureIndex = int
"""Index of the quadrature component (0=I, 1=Q)."""
ComponentId = tuple[UUID4, QuadratureIndex]
"""Index of an individual pulse component.

Each pulse is labeled by a unique identifier, but it is associated to two components,
corresponding to its quadratures (I, Q).
"""
WaveformIndex = int
"""Index of the memory block containing the given waveform samples."""
WaveformIndices = dict[ComponentId, tuple[WaveformIndex, int]]
"""Map pulses' components to waveforms memory indices, and related duration."""


class Waveform(Model):
    data: Annotated[ArrayList, AfterValidator(lambda a: a.astype(float))]
    index: int


class WaveformSpec(Model):
    waveform: Waveform
    duration: int


def _pulse(event: Union[Pulse, Readout], amplitude_swept: bool) -> Pulse:
    """Extract pulse from event.

    Accounts for nested pulses within :class:`Readout` operations.

    Also reset amplitude to 1 for swept pulses, to avoid doubly scaling their amplitude
    statically and in the sweeper implementation.

    .. todo:

        Lift amplitude rescaling to the generic platform sequence pre-processing, since
        driver-independent.

    """
    update = {"amplitude": 1.0} if amplitude_swept else {}
    return (event.probe if isinstance(event, Readout) else event).model_copy(
        update=update
    )


def _waveform(
    pulse: Pulse,
    component: str,
    sampling_rate: float,
    duration: float | None = None,
    index: int = 0,
) -> WaveformSpec:
    duration_ = pulse.duration if duration is None else duration
    update = {"duration": duration_}
    return WaveformSpec(
        waveform=Waveform(
            data=getattr(pulse.model_copy(update=update), component)(sampling_rate),
            index=index,
        ),
        duration=int(duration_),
    )


def _deduplicate(
    pulses: Sequence[Pulse],
) -> tuple[list[Pulse], npt.NDArray[np.int_]]:
    """Deduplicate non-swept pulses

    The reason swept pulses are not deduplicated is that they are swept over duration so
    there will be no duplicates. It is still possible that a swept pulse is the same as
    a non-swept pulse but this is not a prominent enough use-case to justify accounting
    for it.

    Args:
        pulses: A sequence of Pulse objects to deduplicate.

    Returns:
        A tuple containing:
            - list[Pulse]: A list of unique pulses in order of first appearance.
            - npt.NDArray[np.int_]: An array of indices mapping each original pulse to
              its corresponding index in the deduplicated list.
    """
    hashes = np.array([hash(p) for p in pulses])
    _, unique_idx, inverse_idx = np.unique(
        hashes, return_index=True, return_inverse=True
    )
    unique_pulses = np.array(pulses)[unique_idx]
    return list(unique_pulses), inverse_idx


def waveforms(
    sequence: Iterable[PulseLike],
    sampling_rate: float,
    amplitude_swept: set[PulseId],
    duration_swept: dict[PulseLike, Sweeper],
) -> tuple[dict[WaveformIndex, WaveformSpec], WaveformIndices]:
    pulses = [
        _pulse(e, e.id in amplitude_swept)
        for e in sequence
        if isinstance(e, (Pulse, Readout))
    ]

    pulses_not_swept = [p for p in pulses if p not in duration_swept]
    pulses_swept = [
        (_pulse(p, p.id in amplitude_swept), duration_swept[p])
        for p in duration_swept
        if isinstance(p, (Pulse, Readout))
    ]

    unique_pulses, inverse_idx = _deduplicate(pulses_not_swept)

    # the ids for the swept pulses start counting from `static` since up to here we need
    # two indices (i and q) for each unique non-swept pulse
    static = 2 * len(unique_pulses)

    # mapping from integer to unique WaveformSpec
    waveform_specs: dict[int, WaveformSpec] = {  # non-swept
        2 * k + ch: _waveform(
            pulse,
            comp,
            sampling_rate,
            index=k * 2 + ch,
        )
        for k, pulse in enumerate(unique_pulses)
        for ch, comp in enumerate(("i", "q"))
    } | {  # swept
        static + 2 * k + ch: _waveform(
            pulse,
            comp,
            sampling_rate,
            duration=cast(float, duration),
            index=static + 2 * k + ch,
        )
        for k, (pulse, sweep) in enumerate(pulses_swept)
        for duration in np.arange(*sweep.irange)
        for ch, comp in enumerate(("i", "q"))
    }

    # mapping that associate each element in the full list of pulses identified by
    # (UUID, i or q) to an integer that can be associated with a WaveformSpec through
    # waveform_specs
    indices_map: WaveformIndices = {  # non-swept
        (pulse.id, ch): (inv * 2 + ch, int(pulse.duration))
        for inv, pulse in zip(inverse_idx, pulses_not_swept)
        for ch, _ in enumerate(("i", "q"))
    } | {  # swept
        (pulse.id, 2 * i + ch): (static + 2 * k + ch, int(duration))
        for k, (pulse, sweep) in enumerate(pulses_swept)
        for i, duration in enumerate(np.arange(*sweep.irange))
        for ch, _ in enumerate(("i", "q"))
    }

    return waveform_specs, indices_map
