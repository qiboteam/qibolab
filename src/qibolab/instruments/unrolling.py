"""Utilities for sequence unrolling.

May be reused by different instruments.
"""


def batch_max_readout(sequences, max_measurements):
    """Split a list of sequences to batches with a maximum number of readout pulses.

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
