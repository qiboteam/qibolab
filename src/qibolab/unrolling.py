"""Utilities for sequence unrolling.

May be reused by different instruments.
"""

from dataclasses import asdict, dataclass, field, fields

from more_itertools import chunked

from .pulses import PulseSequence


def batch_max_sequences(sequences, max_size):
    """Split a list of sequences to batches with a maximum number of sequences
    in each.

    Args:
        sequences (list): List of :class:`qibolab.pulses.PulseSequence` objects.
        max_size (int): Maximum number of sequences in a single batch.
    """
    return chunked(sequences, max_size)


def batch_max_duration(sequences, max_duration):
    """Split a list of sequences to batches with a trying to get an waveform
    memory for the control pulses estimate using the duration of the sequence.

    Args:
        sequences (list): List of :class:`qibolab.pulses.PulseSequence` objects.
        max_duration (int): Maximum number of readout pulses in a single batch.
    """
    batch_duration, batch = 0, []
    for sequence in sequences:
        duration = sequence.duration - sequence.ro_pulses.duration
        if duration + batch_duration > max_duration:
            yield batch
            batch_duration, batch = duration, [sequence]
        else:
            batch.append(sequence)
            batch_duration += duration
    yield batch


def batch_max_readout(sequences, max_measurements):
    """Split a list of sequences to batches with a maximum number of readout
    pulses in each.

    Useful for sequence unrolling on instruments that support a maximum number of readout pulses
    in a single sequence due to memory limitations.

    Args:
        sequences (list): List of :class:`qibolab.pulses.PulseSequence` objects.
        max_measurements (int): Maximum number of readout pulses in a single batch.
    """

    batch_measurements, batch = 0, []
    for sequence in sequences:
        nmeasurements = len(sequence.ro_pulses)
        if nmeasurements + batch_measurements > max_measurements:
            yield batch
            batch_measurements, batch = nmeasurements, [sequence]
        else:
            batch.append(sequence)
            batch_measurements += nmeasurements
    yield batch


def _waveform(sequence: PulseSequence):
    # TODO: deduplicate pulses (Not yet as drivers may not support it yet)
    # TODO: count Rectangular and delays separately (Zurich Instruments supports this)
    # TODO: Any constant part of a pulse should be counted only once (Zurich Instruments supports this)
    # TODO: check if readout duration is faithful for the readout pulse (I would only check the control pulses)
    return sequence.duration - sequence.ro_pulses.duration


def _readout(sequence: PulseSequence):
    return len(sequence.ro_pulses)


def _instructions(sequence: PulseSequence):
    return len(sequence)


@dataclass(frozen=True, order=False)
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

    def __ge__(self, other: "Bounds") -> bool:
        return all(
            getattr(self, f.name) >= getattr(other, f.name) for f in fields(self)
        )


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
