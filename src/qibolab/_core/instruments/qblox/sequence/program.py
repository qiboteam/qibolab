from collections.abc import Sequence

from qibolab._core.execution_parameters import AveragingMode, ExecutionParameters
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses.pulse import PulseId, PulseLike
from qibolab._core.sweeper import ParallelSweepers

from ..q1asm.ast_ import (
    Block,
    Instruction,
    Line,
    Move,
    Program,
    Stop,
    UpdParam,
    Wait,
    WaitSync,
)
from .acquisition import AcquisitionSpec, MeasureId
from .experiment import experiment
from .finalize import DEFAULT_PIPELINE, transform
from .loops import LoopSpec, Registers, loop, loops
from .sweepers import (
    IndexedParams,
    LoopParams,
    Param,
    params,
    params_reshape,
    sweep_sequence,
    update_instructions,
)
from .waveforms import PulseRealization

__all__ = ["Program", "twpa_program"]


def setup(
    loops: Sequence[LoopSpec],
    params: list[Param],
    channel: set[ChannelId],
    pulses: set[PulseId],
) -> Block:
    """Build preparation phase. Ending with synchronization.

    This will set up all the registers used for iterations, parameters
    updates (sweeps), and to track the bin index where to save the
    results.

    The initial values for channel-wide parameters is also set.

    It is guaranteed that the final instruction of this block will
    trigger the synchronization among modules, with no other instruction
    trailing.
    """
    return (
        Registers.init_zero_registers()
        + [
            Line(
                instruction=Move(source=lp.length, destination=lp.reg),
                comment=f"init {lp.description} counter",
            )
            for lp in loops
        ]
        + [
            Line(
                instruction=Move(source=p.start, destination=p.reg),
                comment=f"init {p.description}",
            )
            for p in params
            if p.channel in channel or p.pulse in pulses
        ]
        + [
            inst
            for p in params
            if p.channel in channel
            for inst in update_instructions(p.role, p.reg)
        ]
    )


def finalization() -> list[Instruction]:
    """Finalize.

    Currently only stopping the sequencer.
    """
    return [Stop()]


def _allocate_registers(
    options: ExecutionParameters,
    sweepers: list[ParallelSweepers],
    channel: set[ChannelId],
    pulses: set[PulseId],
) -> tuple[Sequence[LoopSpec], list[Param], IndexedParams]:
    """Allocate registers for loops and swept parameters."""
    assert options.nshots is not None
    loops_ = loops(
        sweepers,
        options.nshots,
        inner_shots=options.averaging_mode is AveragingMode.SEQUENTIAL,
    )
    params_ = params(
        sweepers,
        allocated=max(lp.reg.number for lp in loops_),
        pulses=pulses,
        channels=channel,
    )
    # sequencers with no parameter in a loop still need an (empty) entry for it
    indexed_params = {
        lp.id: LoopParams(channel=[], pulse=[]) for lp in loops_ if lp.id is not None
    } | params_reshape(params_)
    return loops_, params_, indexed_params


def _assemble(
    experiment_: list[Instruction],
    loops_: Sequence[LoopSpec],
    params_: list[Param],
    indexed_params: IndexedParams,
    options: ExecutionParameters,
    relaxation_time: int,
    channel: set[ChannelId],
    pulses: set[PulseId],
) -> Program:
    """Wrap the experiment in the loops machinery, and finalize the program."""
    block: Block = [
        *setup(loops_, params_, channel, pulses),
        *loop(
            experiment_,
            loops_,
            indexed_params,
            relaxation_time,
            options.averaging_mode is AveragingMode.SINGLESHOT,
            channel,
            pulses,
        ),
        *finalization(),
    ]
    return Program.from_elements(transform(block, DEFAULT_PIPELINE))


def program(
    sequence: Sequence[PulseLike],
    pulse_realization: PulseRealization,
    acquisitions: dict[MeasureId, AcquisitionSpec],
    options: ExecutionParameters,
    sweepers: list[ParallelSweepers],
    channel: set[ChannelId],
    padding: int,
    merged_vzs: bool,
) -> Program:
    """Generate sequencer program."""
    assert options.relaxation_time is not None
    loops_, params_, indexed_params = _allocate_registers(
        options, sweepers, channel, {p.id for p in sequence}
    )
    sweepseq = sweep_sequence(
        sequence, [p for v in indexed_params.values() for p in v.pulse]
    )
    experiment_ = [
        *experiment(sweepseq, pulse_realization, acquisitions, merged_vzs),
        # Enforce a minimum wait of 4 ns corresponding to one clock cycle
        Wait(duration=max(padding, 4)),
    ]
    return _assemble(
        experiment_,
        loops_,
        params_,
        indexed_params,
        options,
        options.relaxation_time,
        channel,
        {p[0].id for p in sweepseq},
    )


def twpa_program(
    options: ExecutionParameters,
    sweepers: list[ParallelSweepers],
    channel: set[ChannelId],
    duration: int,
) -> Program:
    """Generate sequencer program for swept TWPA channel."""
    loops_, params_, indexed_params = _allocate_registers(
        options, sweepers, channel, set()
    )
    experiment_ = [
        UpdParam(duration=4),
        WaitSync(duration=4),
        # Enforce a minimum wait of 4 ns corresponding to one clock cycle
        Wait(duration=max(duration, 4)),
    ]
    return _assemble(
        experiment_, loops_, params_, indexed_params, options, 0, channel, set()
    )
