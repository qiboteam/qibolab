"""Batching utilities for Qblox cluster sequence execution."""

from qibolab import Delay
from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.identifier import ChannelId
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers

from .utils import per_shot_memory
from .validate import cluster_memory_limits


def _init_batch() -> dict[str, int]:
    """Helper function to initialise the batch tracking variables."""
    return {
        "acq_memory": 0,
        "acq_number": 0,
        # an offset number of lines that is always there regardless of the number of
        # pulses played.
        # WARNING: this was determined empirically from the lines for QRM and QCM, so
        # the individual numbers may be lower.
        "qcm_lines": 50,
        "qrm_lines": 50,
    }


def _exceeds_memory_limits(
    batch_memory: dict[str, int], ps_memory: dict[str, int]
) -> bool:
    """Return whether adding ``ps_memory`` to ``batch_memory`` exceeds the
    ``cluster_memory_limits``.
    """
    return any(
        batch_memory[key] + ps_memory[key] > limit
        for key, limit in cluster_memory_limits.items()
    )


def _split_sequences_by_memory_limits(
    sequences: list[PulseSequence],
    sweepers: list[ParallelSweepers],
    options: ExecutionParameters,
    drive_channels: set[ChannelId],
    acquisition_channels: set[ChannelId],
) -> list[list[PulseSequence]]:
    """Split sequences into batches that fit into the cluster memory.

    Returns a list of batches, where each batch is a list of sequences.
    """
    batch: list[PulseSequence] = []
    batch_memory = _init_batch()
    batches: list[list[PulseSequence]] = []
    for ps in sequences:
        ps_memory = {
            "acq_memory": per_shot_memory(ps, sweepers, options),
            "acq_number": len(ps.acquisitions),
            # The factor 1.6 is determined heuristically, for large number of gates
            # and iterations the ratio of ps objects to Lines is approx 1.56
            # WARNING: it was determined by combining QRM and QRC instructions, but the
            # the two types of instructions may have a different factor.
            # TODO: use the number of post-compilation lines
            "qcm_lines": sum(
                1.6 for channelid, _pulse in ps if channelid in drive_channels
            ),
            "qrm_lines": sum(
                1.6 for channelid, _pulse in ps if channelid in acquisition_channels
            ),
        }

        # Check if the pulse sequence on its own exceeds the clusters memory limit
        if _exceeds_memory_limits(_init_batch(), ps_memory):
            raise ValueError(
                "An individual sequence exceeds Qblox cluster memory limits"
            )

        # TODO: track instruction memory usage per module instead of summing across
        # all modules.
        if _exceeds_memory_limits(batch_memory, ps_memory):
            batches.append(batch)
            batch_memory = _init_batch()
            batch = []

        for key, mem in ps_memory.items():
            batch_memory[key] += mem
        batch.append(ps)

    # If the the loop over sequences ended with a non-empty batch, add it to the
    # list of batches.
    if batch:
        batches.append(batch)

    return batches


def _merge_batch_sequences(
    batches: list[list[PulseSequence]],
    options: ExecutionParameters,
) -> list[PulseSequence]:
    """Takes a list of batches (where each batch is a list of PulseSequences) and merges
    all PulseSequences within each batch into a single PulseSequence per batch.
    """
    merged_batches: list[PulseSequence] = []
    for batch in batches:
        batched = batch[0]
        for ps in batch[1:]:
            # Place a Delay with the duration of the relaxation time between each
            # sequence. The pipe operation aligns all channels so we only have to add
            # the Delay to a single channel
            assert options.relaxation_time is not None
            batched |= [(ps[0][0], Delay(duration=options.relaxation_time))]
            batched |= ps
        merged_batches.append(batched)
    return merged_batches


def batch_sequences_by_cluster_memory_limits(
    sequences: list[PulseSequence],
    sweepers: list[ParallelSweepers],
    options: ExecutionParameters,
    drive_channels: set[ChannelId],
    acquisition_channels: set[ChannelId],
) -> list[PulseSequence]:
    """Split sequences into batches that fit into the cluster memory."""
    batches = _split_sequences_by_memory_limits(
        sequences, sweepers, options, drive_channels, acquisition_channels
    )
    return _merge_batch_sequences(batches, options)
