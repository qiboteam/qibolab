from collections.abc import Iterable, Sequence
from itertools import groupby
from typing import Optional, Union, cast

from qibolab._core.execution_parameters import AveragingMode, ExecutionParameters
from qibolab._core.identifier import ChannelId
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers

from ..q1asm.ast_ import (
    Add,
    Instruction,
    Jge,
    Line,
    Loop,
    Move,
    Program,
    Reference,
    Register,
    ResetPh,
    Stop,
    Wait,
    WaitSync,
)
from .acquisition import AcquisitionSpec, MeasureId
from .loops import Loop, Registers
from .sweepers import Param
from .waveforms import WaveformIndices

__all__ = ["Program"]


def setup(
    loops: Sequence[Loop], params: list[Param]
) -> Sequence[Union[Line, Instruction]]:
    """Set up."""
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
                instruction=Move(source=lp[1], destination=lp[0]),
                comment="init "
                + ("shots" if lp[2] is None else sweep_desc(lp[2]))
                + " counter",
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
            if p.channel is not None
            for inst in update_instructions(p.kind, p.start)
        ]  # TODO: condition on the ID of the current channel, since the sequence is being built for it
        + [WaitSync(duration=4)]
    )


IndexedParams = dict[int, tuple[list[Param], list[Param]]]


def _channels_pulses(
    pars: Iterable[tuple[int, Param]],
) -> tuple[list[Param], list[Param]]:
    channels = []
    pulses = []
    for p in pars:
        (channels if p[1].channel is not None else pulses).append(p[1])
    return channels, pulses


def params_reshape(params: Params) -> IndexedParams:
    """Split parameters related to channels and pulses.

    Moreover, it reorganize them by loop, to group the updates.
    """

    return {
        key: _channels_pulses(pars) for key, pars in groupby(params, key=lambda t: t[0])
    }


START = "start"
SHOTS = "shots"


def iteration_end(relaxation_time: int) -> Sequence[Line]:
    return [
        Line(instruction=Wait(duration=relaxation_time), comment="relaxation"),
        Line(instruction=ResetPh(), comment="phase reset"),
        Line(
            instruction=Add(
                a=Registers.bin.value, b=1, destination=Registers.bin.value
            ),
            comment="bin increment",
        ),
    ]


def loop_machinery(
    loops: Loops, params: IndexedParams, singleshot: bool, channel: ChannelId
) -> Sequence[Union[Line, Instruction]]:
    def shots(marker: Optional[int]) -> bool:
        return marker is None and not singleshot

    return [
        i_
        for lp in loops
        for i_ in (
            [
                Line(
                    instruction=Jge(
                        a=Registers.shots.value, b=1, address=Reference(label=SHOTS)
                    ),
                    comment="skip bin reset - advance both counters",
                ),
                Line(
                    instruction=Move(
                        source=Registers.bin_reset.value,
                        destination=Registers.bin.value,
                    ),
                    comment="shots average: reset bin counter",
                ),
            ]
            if shots(lp[2])
            else []
        )
        + [
            *(
                (
                    line
                    for block in (
                        (
                            Line(
                                instruction=Add(a=p.reg, b=p.step, destination=p.reg),
                                comment=f"increment {p.description}",
                            ),
                            *(
                                update_instructions(p.kind, p.reg)
                                if p.description is not None and p.channel == channel
                                else ()
                            ),
                        )
                        for p in (params[lp[2]][0] + params[lp[2]][1])
                    )
                    for line in block
                )
                if lp[2] is not None
                else ()
            ),
            Line(
                instruction=Loop(a=lp[0], address=Reference(label=START)),
                comment="loop over "
                + ("shots" if lp[2] is None else sweep_desc(lp[2])),
                label=SHOTS if shots(lp[2]) else None,
            ),
            Move(source=lp[1], destination=lp[0]),
        ]
    ][:-1]


def loop(
    loops: Loops,
    params: IndexedParams,
    experiment: list[Instruction],
    relaxation_time: int,
    singleshot: bool,
    channel: ChannelId,
) -> Sequence[Union[Line, Instruction]]:
    end = cast(list, iteration_end(relaxation_time))
    machinery = cast(list, loop_machinery(loops, params, singleshot, channel))
    main = experiment + end + machinery

    return [
        (
            Line(instruction=main[0], label=START)
            if isinstance(main[0], Instruction)
            else Line(
                instruction=main[0].instruction, comment=main[0].comment, label=START
            )
        )
    ] + main[1:]


def finalization() -> list[Instruction]:
    """Finalize."""
    return [Stop()]


MAX_WAIT = 2**16 - 1


def long_wait(duration: int) -> list[Line]:
    n = 60
    iterations = duration // MAX_WAIT
    remainder = duration % MAX_WAIT
    register = Register(number=n)
    label = f"wait-{n}"
    return [Line.instr(Wait(duration=remainder))] + [
        Line.instr(Move(source=iterations, destination=register)),
        Line(instruction=Wait(duration=MAX_WAIT), label=label),
        Line.instr(Loop(a=register, address=Reference(label=label))),
    ]


def decompose(line: Line) -> list[Line]:
    if not isinstance(line.instruction, Wait):
        return [line]
    duration = line.instruction.duration
    if not isinstance(duration, int) or duration <= MAX_WAIT:
        return [line]
    wait = long_wait(duration)
    return [
        Line(instruction=wait[0].instruction, label=line.label, comment=line.comment)
    ] + wait[1:]


def transpile(prog: Program) -> Program:
    return Program(
        elements=[
            el
            for oel in prog.elements
            for el in (decompose(oel) if isinstance(oel, Line) else [oel])
        ]
    )


def program(
    sequence: PulseSequence,
    waveforms: WaveformIndices,
    acquisitions: dict[MeasureId, AcquisitionSpec],
    options: ExecutionParameters,
    sweepers: list[ParallelSweepers],
    sampling_rate: float,
    channel: ChannelId,
) -> Program:
    assert options.nshots is not None
    assert options.relaxation_time is not None

    loops_ = loops(
        sweepers,
        options.nshots,
        inner_shots=options.averaging_mode is AveragingMode.SEQUENTIAL,
    )
    params_ = params(sweepers, allocated=max(lp[0].number for lp in loops_))
    indexed_params = params_reshape(params_)
    sweepseq = sweep_sequence(
        sequence, [p for v in indexed_params.values() for p in v[1]]
    )

    return transpile(
        Program(
            elements=[
                el if isinstance(el, Line) else Line.instr(el)
                for block in [
                    setup(loops_, [p for _, p in params_]),
                    loop(
                        loops_,
                        indexed_params,
                        execution(sweepseq, waveforms, acquisitions, sampling_rate),
                        options.relaxation_time,
                        options.averaging_mode is AveragingMode.SINGLESHOT,
                        channel,
                    ),
                    finalization(),
                ]
                for el in block
            ]
        )
    )
