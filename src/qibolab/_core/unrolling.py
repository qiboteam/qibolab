"""Utilities for sequence unrolling.

May be reused by different instruments.
"""

from collections import defaultdict
from functools import total_ordering
from typing import Annotated, Literal

from pydantic.fields import FieldInfo

from .components.configs import Config
from .pulses import Delay, Pulse
from .pulses.envelope import Rectangular
from .sequence import PulseSequence


def _waveform(sequence: PulseSequence):
    # TODO: deduplicate pulses (Not yet as drivers may not support it yet)
    # TODO: VirtualZ deserves a separate handling
    # TODO: any constant part of a pulse should be counted only once (Zurich Instruments supports this)
    # TODO: handle multiple qubits or do all devices have the same memory for each channel ?
    return sum(
        (
            (pulse.duration if not isinstance(pulse.envelope, Rectangular) else 1)
            if isinstance(pulse, Pulse)
            else 1
        )
        for _, pulse in sequence
    )


def _readout(sequence: PulseSequence):
    # TODO: Do we count 1 readout per pulse or 1 readout per multiplexed readout ?
    return len(sequence.acquisitions)


def _instructions(sequence: PulseSequence):
    return len(sequence)


@total_ordering
class Bounds(Config):
    """Instument memory limitations proxies."""

    kind: Literal["bounds"] = "bounds"

    waveforms: Annotated[int, {"count": _waveform}]
    """Waveforms estimated size."""
    readout: Annotated[int, {"count": _readout}]
    """Number of readouts."""
    instructions: Annotated[int, {"count": _instructions}]
    """Instructions estimated size."""

    @classmethod
    def update(cls, sequence: PulseSequence):
        up = {}
        for name, info in cls._entries().items():
            up[name] = info.metadata[0]["count"](sequence)

        return cls(**up)

    def __add__(self, other: "Bounds") -> "Bounds":
        """Sum bounds element by element."""
        new = {}
        for (k, x), (_, y) in zip(
            self.model_dump().items(), other.model_dump().items()
        ):
            if k in type(self)._entries():
                new[k] = x + y

        return type(self)(**new)

    def __gt__(self, other: "Bounds") -> bool:
        """Define ordering as exceeding any bound."""
        return any(getattr(self, f) > getattr(other, f) for f in type(self)._entries())

    @classmethod
    def _entries(cls) -> dict[str, FieldInfo]:
        return {
            n: f
            for n, f in cls.model_fields.items()
            if "count" in next(iter(f.metadata), {})
        }


def batch(sequences: list[PulseSequence], bounds: Bounds):
    """Split a list of sequences to batches.

    Takes into account the various limitations throught the mechanics defined in
    :class:`Bounds`, and the numerical limitations specified by the `bounds` argument.
    """
    counters = Bounds(waveforms=0, readout=0, instructions=0)
    batch = []
    for sequence in sequences:
        update = Bounds.update(sequence)
        if counters + update > bounds:
            yield batch
            counters, batch = update, [sequence]
        else:
            batch.append(sequence)
            counters += update
    yield batch


def unroll_sequences(
    sequences: list[PulseSequence], relaxation_time: int
) -> tuple[PulseSequence, dict[int, list[int]]]:
    """Unrolls a list of pulse sequences to a single sequence.

    The resulting sequence may contain multiple measurements.

    `relaxation_time` is the time in ns to wait for the qubit to relax between playing
    different sequences.

    It returns both the unrolled pulse sequence, and the map from original readout pulse
    serials to the unrolled readout pulse serials. Required to construct the results
    dictionary that is returned after execution.
    """
    total_sequence = PulseSequence()
    readout_map = defaultdict(list)
    for sequence in sequences:
        total_sequence.concatenate(sequence)
        # TODO: Fix unrolling results
        for _, acq in sequence.acquisitions:
            readout_map[acq.id].append(acq.id)

        length = sequence.duration + relaxation_time
        for channel in sequence.channels:
            delay = length - sequence.channel_duration(channel)
            total_sequence.append((channel, Delay(duration=delay)))

    return total_sequence, readout_map
