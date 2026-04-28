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


def _merge_batched_sequences(
    batches: list[list[PulseSequence]],
    options: ExecutionParameters,
) -> list[PulseSequence]:
    """Takes a list of batches (where each batch is a list of PulseSequences) and merges
    all PulseSequences within each batch into a single PulseSequence per batch.
    """
    merged_batches: list[PulseSequence] = []
    for batch in batches:
        # copy to avoid mutating the input pulse sequence
        batched = batch[0].copy()
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
    qcm_channels: set[ChannelId],
    qrm_channels: set[ChannelId],
) -> list[PulseSequence]:
    """
    Split sequences into batches using an approximate best-fit algorithm: for each
    sequence, place it in the batch where it leaves the least remaining space. If no
    batch fits, start a new batch.
    """
    batches: list[list[PulseSequence]] = []
    batches_memory: list[dict[str, int]] = []

    for pulse_sequence in sequences:
        pulse_sequence_memory = {
            "acq_memory": per_shot_memory(pulse_sequence, sweepers, options),
            "acq_number": len(pulse_sequence.acquisitions),
            # The factor 1.6 is determined heuristically, for large number of gates
            # and iterations the ratio of ps objects to Lines is approx 1.56
            # WARNING: it was determined by combining QRM and QRC instructions, but the
            # the two types of instructions may have a different factor.
            # TODO: use the number of post-compilation lines
            "qcm_lines": sum(
                1.6 for channelid, _pulse in pulse_sequence if channelid in qcm_channels
            ),
            "qrm_lines": sum(
                1.6 for channelid, _pulse in pulse_sequence if channelid in qrm_channels
            ),
        }

        # Check if the pulse sequence on its own exceeds the clusters memory limit
        if _exceeds_memory_limits(_init_batch(), pulse_sequence_memory):
            raise ValueError(
                "An individual sequence exceeds Qblox cluster memory limits"
            )

        # Find the (approximate) best-fit batch
        best_batch_idx = None
        min_max_remaining = 1.0
        for batch_idx, batch_memory in enumerate(batches_memory):
            # Check if adding ps_memory to this batch would exceed limits
            if _exceeds_memory_limits(batch_memory, pulse_sequence_memory):
                continue
            # Compute the max remaining space in the batch after adding the pulse
            # sequence
            max_remaining = max(
                (
                    cluster_memory_limits[key]
                    - (batch_memory[key] + pulse_sequence_memory[key])
                )
                / cluster_memory_limits[key]
                for key in cluster_memory_limits
            )
            if max_remaining < min_max_remaining:
                min_max_remaining = max_remaining
                best_batch_idx = batch_idx

        if best_batch_idx is not None:
            # Place the pulse sequence in the best-fit batch
            for key in pulse_sequence_memory:
                batches_memory[best_batch_idx][key] += pulse_sequence_memory[key]
            batches[best_batch_idx].append(pulse_sequence)
        else:
            # Start a new batch
            batches.append([pulse_sequence])
            new_mem = _init_batch()
            for key in pulse_sequence_memory:
                new_mem[key] += pulse_sequence_memory[key]
            batches_memory.append(new_mem)

    return _merge_batched_sequences(batches, options)
