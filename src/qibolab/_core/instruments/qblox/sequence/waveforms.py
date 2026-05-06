from collections.abc import Iterable, Sequence
from itertools import count
from typing import Annotated, Union

import numpy as np
import numpy.typing as npt
from pydantic import UUID4, AfterValidator

from qibolab._core.pulses import Pulse, PulseId, PulseLike, Readout
from qibolab._core.serialize import ArrayList, Model
from qibolab._core.sweeper import Sweeper

__all__ = []

QuadratureIndex = int
"""Index of the quadrature component (even=I, odd=Q)."""
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


def _deduplicate_pulses(
    pulses: Sequence[Pulse],
) -> tuple[list[Pulse], npt.NDArray[np.int_]]:
    """Deduplicate non-swept pulses based on waveform-determining fields.

    The reason swept pulses are not deduplicated is that they are swept over duration so
    there will be no duplicates. It is still possible that a swept pulse is the same as
    a non-swept pulse but this is not a prominent enough use-case to justify accounting
    for it.

    Deduplication is based on (duration, amplitude, envelope) only. The
    ``relative_phase`` field is intentionally excluded because it does not affect the
    uploaded waveform samples since it is handled via ``set_ph_delta`` in the Q1ASM
    program.

    Returns:
        A tuple containing:
            - list[Pulse]: A list of unique pulses.
            - npt.NDArray[np.int_]: Array of indices mapping each original pulse to its
              corresponding index in the deduplicated list.
    """
    hashes = np.array([(p.duration, p.amplitude, hash(p.envelope)) for p in pulses])
    _, unique_idx, inverse_idx = np.unique(
        hashes, axis=0, return_index=True, return_inverse=True
    )
    unique_pulses = np.array(pulses)[unique_idx]
    return list(unique_pulses), inverse_idx


def _deduplicate_waveforms(
    waveforms: list[WaveformSpec],
) -> tuple[list[WaveformSpec], list[WaveformIndex]]:
    """Deduplicate waveforms by sampled waveform arrays.

    Returns:
        A tuple containing:
            - list[WaveformSpec]: The unique waveforms re-indexed from 0.
            - list[WaveformIndex]: Mapping from each original waveform
              index to its deduplicated waveform index.
    """

    waveform_arrays = np.array(
        [waveform.waveform.data.tobytes() for waveform in waveforms]
    )
    _, unique_idx, inverse_idx = np.unique(
        waveform_arrays, return_index=True, return_inverse=True
    )

    deduplicated = [
        waveforms[orig_index].model_copy(
            update={
                "waveform": waveforms[orig_index].waveform.model_copy(
                    update={"index": new_index}
                )
            }
        )
        for new_index, orig_index in enumerate(unique_idx)
    ]

    return deduplicated, inverse_idx.astype(int).tolist()


def waveforms(
    sequence: Iterable[PulseLike],
    sampling_rate: float,
    amplitude_swept: set[PulseId],
    duration_swept: dict[PulseId, Sweeper],
) -> tuple[dict[WaveformIndex, WaveformSpec], WaveformIndices]:
    """Build the waveform memory map and pulse-component index map for a sequence.

    1. Split pulses into non-swept and duration-swept groups. Amplitude-swept pulses
       have their amplitude reset to 1 so the sequencer scales them at runtime.
    2. Deduplicate non-swept pulses structurally (by duration, amplitude, and envelope
       hash) to avoid uploading identical waveforms for repeated pulses.
    3. Sample waveforms: each unique pulse produces two entries (I, Q);
    4. Deduplicate the sampled arrays a second time. This is mainly to catch cases where
       unique envelopes share the same, non-unique, I or Q component.
    5. Construct ``indices_map`` mapping ``(pulse UUID, quadrature)`` to the final
       deduplicated memory index and the corresponding duration.

    This is the workflow represented as a diagram::

                  pulses          ┃    pulses deduplicated    ┃         waveforms         ┃   waveforms deduplicated
        ━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━
             ┏━━━┳━━━┓            ┃                           ┃                           ┃
             ┃   ┃   ┃ ━━━━━━━━━━━╋━━━━━┓     ┏━━━┳━━━┓       ┃           ┏━━━┓           ┃            ┏━━━┓
             ┗━━━┻━━━┛            ┃     ┗━━━▶ ┃   ┃   ┃ ━━━━━━╋━━━━━┓     ┃   ┃  ━━━━━━━━━╋━━━━━━━━━▶  ┃   ┃
             ┏━━━┳━━━┓            ┃     ┏━━━▶ ┗━━━┻━━━┛       ┃     ┗━━━▶ ┣━━━┫           ┃            ┣━━━┫
             ┃   ┃   ┃ ━━━━━━━━━━━╋━━━━━┛                     ┃           ┃   ┃  ━━━━━━━━━╋━━━━━━━━━▶  ┃   ┃
             ┗━━━┻━━━┛            ┃                           ┃           ┗━━━┛           ┃            ┣━━━┫
             ┏━━━┳━━━┓            ┃           ┏━━━┳━━━┓       ┃                           ┃     ┏━━━▶  ┃   ┃
             ┃   ┃   ┃ ━━━━━━━━━━━╋━━━━━━━━━▶ ┃   ┃   ┃ ━━━━━━╋━━━━━┓     ┏━━━┓           ┃     ┃      ┗━━━┛
             ┗━━━┻━━━┛            ┃           ┗━━━┻━━━┛       ┃     ┃     ┃   ┃  ━━━━━━━━━╋━━━━━┨
                                  ┃                           ┃     ┗━━━▶ ┣━━━┫           ┃     ┃
                ...               ┃              ...          ┃           ┃   ┃  ━━━━━━━━━╋━━━━━┛
                                  ┃                           ┃           ┗━━━┛           ┃
                                  ┃                           ┃                           ┃
                                 inv                       2x + 0/1        ...      orig_to_dedup
                             ━━━━━╋━━━━▶                ━━━━━━╋━━━━━━▶           ━━━━━━━━━╋━━━━━━━━▶
                                  ┃                           ┃                           ┃
                                  ┃                           ┃                           ┃
                   ━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━ indices_map ━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━▶
    """

    pulses_not_swept: list[Pulse] = []
    pulses_swept: list[tuple[Pulse, Sweeper]] = []
    for p in sequence:
        if isinstance(p, (Pulse, Readout)):
            if p.id in duration_swept:
                pulses_swept.append(
                    (_pulse(p, p.id in amplitude_swept), duration_swept[p.id])
                )
            else:
                pulses_not_swept.append(_pulse(p, p.id in amplitude_swept))

    unique_pulses, inverse_idx = _deduplicate_pulses(pulses_not_swept)

    # setting a counter for non-swept waveforms
    counter = count()
    # mapping non-swept waveforms from integer to unique WaveformSpec
    non_swept_waveforms: list[WaveformSpec] = [
        _waveform(
            pulse,
            comp,
            sampling_rate,
            index=next(counter),
        )
        for pulse in unique_pulses
        for comp in ("i", "q")
    ]

    # perform deduplication of non-swept waveforms based on their sampled arrays, this is
    # necessary _deduplicate_pulses only deduplicates non-unique Pulses, but not
    # non-unique I and Q components.
    deduplicated_waveforms, orig_to_deduplicated = _deduplicate_waveforms(
        non_swept_waveforms
    )

    # the ids for the swept pulses start counting from `static` since up to here we need
    # two indices (i and q) for each unique non-swept pulse
    static = len(deduplicated_waveforms)
    # setting the counter to the number of deduplicated waveforms
    counter = count(static)
    # mapping swept waveforms from integer to unique WaveformSpec
    swept_waveforms: list[WaveformSpec] = [
        _waveform(
            pulse,
            comp,
            sampling_rate,
            duration=duration,
            index=next(counter),
        )
        for pulse, sweep in pulses_swept
        for duration in np.arange(*sweep.irange)
        for comp in ("i", "q")
    ]

    deduplicated_waveforms += swept_waveforms

    # mapping that associate each element in the full list of pulses identified by
    # (UUID, i or q) to an integer that can be associated with a WaveformSpec through
    # waveforms
    indices_map: WaveformIndices = {
        (pulse.id, iq_idx): (
            # here we are using the inverse index to map back to the original list of waveforms;
            # the waveform list used to generate the `orig_to_deduplicated`
            # mapping is ordered pulse-wise, hence I-Q components of the same pulse are
            # adjacent, that's why we can use `inv * 2 + iq_idx` to get the correct index
            orig_to_deduplicated[inv * 2 + iq_idx],
            int(pulse.duration),
        )
        for inv, pulse in zip(inverse_idx, pulses_not_swept)
        for iq_idx, _ in enumerate(("i", "q"))
    }

    # this offset is necessary since we want to keep track of the index for swept waveforms
    # while we are resetting the counter for every swept pulse
    offset = static
    for pulse, sweep in pulses_swept:
        # resetting the counter for every spwept pulse
        counter = count()
        indices_map |= {  # swept
            (pulse.id, (idx := next(counter))): (
                idx + offset,
                int(duration),
            )
            for duration in np.arange(*sweep.irange)
            for _ in ("i", "q")
        }
        offset += next(counter)

    return dict(enumerate(deduplicated_waveforms)), indices_map
