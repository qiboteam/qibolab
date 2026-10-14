"""Batching utilities for Qblox cluster sequence execution."""

from collections import Counter
from collections.abc import Set as AbstractSet

from qibolab import Delay
from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import PulseId
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers

from .limits import SEQUENCER_REGISTERS, cluster_memory_limits
from .sequence.asm import Registers
from .sequence.sequence import _effective_channels
from .sequence.sweepers import _registers
from .utils import per_shot_memory


def _pulse_sweep_register_counts(
    sweepers: list[ParallelSweepers],
) -> dict[PulseId, int]:
    """Number of registers required by pulse sweep parameters, for each pulse.

    Counts follow what is allocated by :func:`.sequence.sweepers.params`, so
    `sweepers` must be normalized: duration sweeps need a different number of
    registers depending on the pulse, hence one pulse per sweeper. Channel
    parameters are counted by :func:`_channel_sweep_register_counts`.
    """
    counts: Counter[PulseId] = Counter()
    for parsweep in sweepers:
        for sweep in parsweep:
            if sweep.pulses:
                nregisters = len(_registers(sweep))
                for pulse in sweep.pulses:
                    counts[pulse.id] += nregisters
    return dict(counts)


def _channel_sweep_register_counts(
    sweepers: list[ParallelSweepers],
) -> dict[ChannelId, int]:
    """Number of registers required by channel sweep parameters, for each channel."""
    counts: Counter[ChannelId] = Counter()
    for parsweep in sweepers:
        for sweep in parsweep:
            if sweep.channels:
                nregisters = len(_registers(sweep))
                for channel in sweep.channels:
                    counts[channel] += nregisters
    return dict(counts)


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


def _swept_pulse_ids_per_channel(
    ps: PulseSequence,
    swept_pulse_ids: AbstractSet[PulseId],
    sweeper_channels: AbstractSet[ChannelId],
) -> dict[ChannelId, set[PulseId]]:
    """Unique swept pulse IDs for the sequencer of each channel.

    Each channel runs its own program, so registers are not shared among channels.
    """
    by_channel = ps.by_channel
    return {
        ch: {
            pulse.id for pulse in by_channel.get(ch, []) if pulse.id in swept_pulse_ids
        }
        for ch in by_channel.keys() | sweeper_channels
    }


def _base_registers_per_channel(
    ps: PulseSequence,
    fixed_register_count: int,
    register_counts_per_channel: dict[ChannelId, int],
) -> dict[ChannelId, int]:
    """Registers required on each channel's sequencer regardless of the swept pulses.

    Besides the fixed registers, those are the registers of the parameters swept on
    the channels the sequencer is associated to (including the *probe* alias of
    acquisition channels). Channels without pulses still need registers when swept.
    """
    by_channel = ps.by_channel
    return {
        ch: fixed_register_count
        + sum(
            register_counts_per_channel.get(c, 0)
            for c in _effective_channels(ch, by_channel.get(ch, []))
        )
        for ch in by_channel.keys() | register_counts_per_channel.keys()
    }


def _raise_error_if_sequence_exceeds_registers(
    registers: dict[ChannelId, int], base_registers: dict[ChannelId, int]
) -> None:
    """Raise error if a single sequence exceeds the sequencer registers."""
    for channelid, number in registers.items():
        requested = base_registers[channelid] + number
        if requested > SEQUENCER_REGISTERS:
            raise ValueError(
                "An individual sequence exceeds the Q1 sequencer registers on channel "
                f"{channelid}: requested={requested}, "
                f"limit={SEQUENCER_REGISTERS}."
            )


def _raise_error_if_sequence_exceeds_memory_limits(ps_memory: dict[str, int]) -> None:
    """Raise error if a single sequence exceeds the cluster memory limits."""
    base_memory = _init_batch()
    fits_in_memory = all(
        base_memory[key] + ps_memory[key] <= limit
        for key, limit in cluster_memory_limits.items()
    )
    if fits_in_memory:
        return

    exceeded_limits = [
        (
            key,
            cluster_memory_limits[key],
            base_memory[key] + ps_memory[key],
        )
        for key in cluster_memory_limits
        if base_memory[key] + ps_memory[key] > cluster_memory_limits[key]
    ]
    details = "\n".join(
        f"- {key}: requested={requested:,}, limit={limit:,}."
        for key, limit, requested in exceeded_limits
    )
    raise ValueError(
        f"An individual sequence exceeds Qblox cluster memory limits:\n{details}"
    )


def _split_sequences_by_memory_limits(
    sequences: list[PulseSequence],
    sweepers: list[ParallelSweepers],
    options: ExecutionParameters,
    qcm_channels: set[ChannelId],
    qrm_channels: set[ChannelId],
) -> list[list[PulseSequence]]:
    """Split sequences into batches that fit into the cluster memory.

    Returns a list of batches, where each batch is a list of sequences.
    """
    # registers are allocated for the loop counters, and for each swept parameter
    register_counts_per_pulse = _pulse_sweep_register_counts(sweepers)
    register_counts_per_channel = _channel_sweep_register_counts(sweepers)
    fixed_register_count = Registers.first_available() + len(sweepers)

    batch: list[PulseSequence] = []
    batch_memory = _init_batch()
    batch_swept_pulse_ids: dict[ChannelId, set[PulseId]] = {}
    batch_base_registers: dict[ChannelId, int] = {}
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
                1.6 for channelid, _pulse in ps if channelid in qcm_channels
            ),
            "qrm_lines": sum(
                1.6 for channelid, _pulse in ps if channelid in qrm_channels
            ),
        }
        # Repeated pulse IDs share parameter registers within and across sequences.
        ps_swept_pulse_ids = _swept_pulse_ids_per_channel(
            ps, register_counts_per_pulse.keys(), register_counts_per_channel.keys()
        )
        ps_registers = {
            ch: sum(register_counts_per_pulse[pid] for pid in pulse_ids)
            for ch, pulse_ids in ps_swept_pulse_ids.items()
        }

        _raise_error_if_sequence_exceeds_memory_limits(ps_memory)
        base_registers = _base_registers_per_channel(
            ps, fixed_register_count, register_counts_per_channel
        )
        _raise_error_if_sequence_exceeds_registers(ps_registers, base_registers)

        # TODO: track instruction memory usage per module instead of summing across
        # all modules.
        fits_in_memory = all(
            batch_memory[key] + ps_memory[key] <= limit
            for key, limit in cluster_memory_limits.items()
        )
        fits_in_registers = all(
            max(batch_base_registers.get(ch, 0), base_registers[ch])
            + sum(
                register_counts_per_pulse[pid]
                for pid in batch_swept_pulse_ids.get(ch, set()) | pulse_ids
            )
            <= SEQUENCER_REGISTERS
            for ch, pulse_ids in ps_swept_pulse_ids.items()
        )
        if not (fits_in_memory and fits_in_registers):
            batches.append(batch)
            batch_memory = _init_batch()
            batch_swept_pulse_ids = {}
            batch_base_registers = {}
            batch = []

        for ch, pulse_ids in ps_swept_pulse_ids.items():
            batch_swept_pulse_ids.setdefault(ch, set()).update(pulse_ids)
            batch_base_registers[ch] = max(
                batch_base_registers.get(ch, 0), base_registers[ch]
            )

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
    """Split sequences into batches that fit into the cluster memory."""
    batches = _split_sequences_by_memory_limits(
        sequences, sweepers, options, qcm_channels, qrm_channels
    )
    return _merge_batch_sequences(batches, options)
