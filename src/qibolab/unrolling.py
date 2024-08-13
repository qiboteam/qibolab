"""Utilities for sequence unrolling.

May be reused by different instruments.
"""

from dataclasses import asdict, dataclass, field, fields
from functools import total_ordering

from .pulses import Pulse, PulseSequence
from .pulses.envelope import Rectangular


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
    return len(sequence.probe_pulses)


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
        """Sum bounds element by element."""
        new = {}
        for (k, x), (_, y) in zip(asdict(self).items(), asdict(other).items()):
            new[k] = x + y

        return type(self)(**new)

    def __gt__(self, other: "Bounds") -> bool:
        """Define ordering as exceeding any bound."""
        return any(getattr(self, f.name) > getattr(other, f.name) for f in fields(self))


def batch(sequences: list[PulseSequence], bounds: Bounds):
    """Split a list of sequences to batches.

    Takes into account the various limitations throught the mechanics defined in
    :cls:`Bounds`, and the numerical limitations specified by the `bounds` argument.
    """
    counters = Bounds(0, 0, 0)
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
