from collections.abc import Iterable, Sequence

from qibolab._core.execution_parameters import AveragingMode, ExecutionParameters
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses.pulse import PulseLike
from qibolab._core.sweeper import ParallelSweepers

from ..q1asm.ast_ import (
    Block,
    Instruction,
    Line,
    Move,
    Program,
    Stop,
    WaitSync,
)
from .acquisition import AcquisitionSpec, MeasureId
from .experiment import experiment
from .loops import LoopSpec, Registers, loop, loops
from .sweepers import Param, params, params_reshape, sweep_sequence, update_instructions
from .transpile import transpile
from .waveforms import WaveformIndices

__all__ = ["Program"]


def setup(loops: Sequence[LoopSpec], params: list[Param], channel: ChannelId) -> Block:
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
        [
            Line(
                instruction=Move(source=0, destination=Registers.bin.value),
                comment="init bin counter",
            ),
            Line(
                instruction=Move(source=0, destination=Registers.bin_reset.value),
                comment="init bin reset",
            ),
        ]
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
        ]
        + [
            inst
            for p in params
            if p.channel == channel
            for inst in update_instructions(p.kind, p.start)
        ]
        + [WaitSync(duration=4)]
    )


def finalization() -> list[Instruction]:
    """Finalize.

    Currently only stopping the sequencer.
    """
    return [Stop()]


def program(
    sequence: Iterable[PulseLike],
    waveforms: WaveformIndices,
    acquisitions: dict[MeasureId, AcquisitionSpec],
    options: ExecutionParameters,
    sweepers: list[ParallelSweepers],
    sampling_rate: float,
    channel: ChannelId,
) -> Program:
    """Generate sequencer program."""
    assert options.nshots is not None
    assert options.relaxation_time is not None

    loops_ = loops(
        sweepers,
        options.nshots,
        inner_shots=options.averaging_mode is AveragingMode.SEQUENTIAL,
    )
    params_ = params(sweepers, allocated=max(lp.reg.number for lp in loops_))
    indexed_params = params_reshape(params_)
    sweepseq = sweep_sequence(
        sequence, [p for v in indexed_params.values() for p in v[1]]
    )
    experiment_ = experiment(sweepseq, waveforms, acquisitions, sampling_rate)
    singleshot = options.averaging_mode is AveragingMode.SINGLESHOT

    return transpile(
        [
            el
            for block in [
                setup(loops_, [p for _, p in params_], channel),
                loop(
                    experiment_,
                    loops_,
                    indexed_params,
                    options.relaxation_time,
                    singleshot,
                    channel,
                ),
                finalization(),
            ]
            for el in block
        ]
    )
