"""Utilities for sequence unrolling.

May be reused by different instruments.
"""

from collections import defaultdict

from .pulses import Delay
from .sequence import PulseSequence


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
