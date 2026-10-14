"""Batching utilities for Qblox cluster sequence execution."""

from qibolab import Delay
from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import PulseId
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers

from .limits import SEQUENCER_REGISTERS, cluster_memory_limits
from .sequence.asm import Registers
from .sequence.sequence import _effective_channels
from .sequence.sweepers import count_channel_params, count_params
from .utils import per_shot_memory


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


def _registers_per_channel(
    ps: PulseSequence, swept: dict[PulseId | None, int]
) -> dict[ChannelId, int]:
    """Registers required by swept pulses, for the sequencer of each channel.

    Each channel runs its own program, so registers are not shared among channels.
    """
    registers: dict[ChannelId, int] = {}
    for channelid, pulse in ps:
        registers[channelid] = registers.get(channelid, 0) + swept.get(pulse.id, 0)
    return registers


def _base_registers_per_channel(
    ps: PulseSequence, fixed: int, channel_swept: dict[ChannelId, int]
) -> dict[ChannelId, int]:
    """Registers required on each channel's sequencer regardless of the swept pulses.

    Besides the `fixed` ones, those are the registers of the parameters swept on the
    channels the sequencer is associated to (including the *probe* alias of acquisition
    channels).
    """
    return {
        ch: fixed + sum(channel_swept.get(c, 0) for c in _effective_channels(ch, seq))
        for ch, seq in ps.by_channel.items()
    }


def _raise_error_if_sequence_exceeds_registers(
    registers: dict[ChannelId, int], base_registers: dict[ChannelId, int]
):
    """Raise error if a single sequence exceeds the sequencer registers."""
    for channelid, number in registers.items():
        requested = base_registers[channelid] + number
        if requested > SEQUENCER_REGISTERS:
            raise ValueError(
                "An individual sequence exceeds the Q1 sequencer registers on channel "
                f"{channelid}: requested={requested}, "
                f"limit={SEQUENCER_REGISTERS}."
            )


def _raise_error_if_sequence_exceeds_memory_limits(ps_memory: dict[str, int]):
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
    swept = count_params(sweepers)
    channel_swept = count_channel_params(sweepers)
    fixed_registers = Registers.first_available() + len(sweepers)

    batch: list[PulseSequence] = []
    batch_memory = _init_batch()
    batch_registers: dict[ChannelId, int] = {}
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
        # the Q1 sequencer has a limited number of registers, and each swept pulse
        # requires its own on the sequencer playing it
        ps_registers = _registers_per_channel(ps, swept)

        _raise_error_if_sequence_exceeds_memory_limits(ps_memory)
        base_registers = _base_registers_per_channel(ps, fixed_registers, channel_swept)
        _raise_error_if_sequence_exceeds_registers(ps_registers, base_registers)

        # TODO: track instruction memory usage per module instead of summing across
        # all modules.
        fits_in_memory = all(
            batch_memory[key] + ps_memory[key] <= limit
            for key, limit in cluster_memory_limits.items()
        )
        fits_in_registers = all(
            max(batch_base_registers.get(ch, 0), base_registers[ch])
            + batch_registers.get(ch, 0)
            + n
            <= SEQUENCER_REGISTERS
            for ch, n in ps_registers.items()
        )
        if not (fits_in_memory and fits_in_registers):
            batches.append(batch)
            batch_memory = _init_batch()
            batch_registers = {}
            batch_base_registers = {}
            batch = []

        for ch, n in ps_registers.items():
            batch_registers[ch] = batch_registers.get(ch, 0) + n
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


def drop_sweeper_pulses_outside_sequence(
    sweepers: list[ParallelSweepers], sequence: PulseSequence
) -> list[ParallelSweepers]:
    """Drop from the sweepers the pulses that are not part of the sequence.

    Registers are allocated for every swept pulse, so, once the sequences have been
    split in batches, each batch should only account for its own pulses. The sweepers
    themselves are kept, since they define the loops (and thus the results' shape).
    """
    ids = {pulse.id for _channel, pulse in sequence}
    return [
        [
            (
                sweeper.model_copy(
                    update={"pulses": [p for p in sweeper.pulses if p.id in ids]}
                )
                if sweeper.pulses is not None
                else sweeper
            )
            for sweeper in parsweep
        ]
        for parsweep in sweepers
    ]
