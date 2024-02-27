"""Utilities for sequence unrolling.

May be reused by different instruments.
"""

from dataclasses import asdict, dataclass, field, fields
from functools import total_ordering

from .pulses import PulseSequence


def _waveform(sequence: PulseSequence):
    # TODO: deduplicate pulses (Not yet as drivers may not support it yet)
    # TODO: count Rectangular and delays separately (Zurich Instruments supports this)
    # TODO: Any constant part of a pulse should be counted only once (Zurich Instruments supports this)
    # TODO: check if readout duration is faithful for the readout pulse (I would only check the control pulses)
    # TODO: Handle multiple qubits or do all devices have the same memory for each channel ?
    return sequence.duration - sequence.ro_pulses.duration


def _readout(sequence: PulseSequence):
    return len(sequence.ro_pulses)


def _instructions(sequence: PulseSequence):
    return len(sequence)


@total_ordering
@dataclass(frozen=True, eq=True)
class Bounds:
    """Instument memory limitations proxies."""

    waveforms: int = field(metadata={"count": _waveform})
    """Waveforms estimated size."""
    readout: int = field(metadata={"count": _readout})
    """Number of readouts."""
    instructions: int = field(metadata={"count": _instructions})
    """Instructions estimated size."""

    @classmethod
    def update(cls, sequence: PulseSequence):
        up = {}
        for f in fields(cls):
            up[f.name] = f.metadata["count"](sequence)

        return cls(**up)

    def __add__(self, other: "Bounds") -> "Bounds":
        new = {}
        for (k, x), (_, y) in zip(asdict(self).items(), asdict(other).items()):
            new[k] = x + y

        return type(self)(**new)

    def __lt__(self, other: "Bounds") -> bool:
        return any(getattr(self, f.name) < getattr(other, f.name) for f in fields(self))


def batch(sequences: list[PulseSequence], bounds: Bounds):
    """Split a list of sequences to batches.

    Takes into account the various limitations throught the mechanics defined in
    :cls:`Bounds`, and the numerical limitations specified by the `bounds` argument.
    """
    counters = Bounds(0, 0, 0)
    batch = []
    for sequence in sequences:
        update_ = Bounds.update(sequence)
        if counters + update_ > bounds:
            yield batch
            counters, batch = update_, [sequence]
        else:
            batch.append(sequence)
            counters += update_
    yield batch
