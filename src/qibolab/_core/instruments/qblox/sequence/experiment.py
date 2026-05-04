from qibolab._core.pulses.pulse import (
    Acquisition,
    Align,
    Delay,
    Pulse,
    Readout,
    VirtualZ,
)
from qibolab._core.sweeper import Parameter

from ..q1asm.ast_ import (
    Acquire,
    Add,
    Block,
    Instruction,
    Line,
    Move,
    Play,
    Register,
    SetPhDelta,
    UpdParam,
    Wait,
    WaitSync,
)
from .acquisition import AcquisitionSpec, MeasureId
from .asm import Registers, convert
from .sweepers import (
    ParameterizedPulse,
    ParamRole,
    SweepSequence,
    reset_instructions,
    update_instructions,
)
from .waveforms import WaveformIndices

__all__ = []


def play_pulse(pulse: Pulse, waveforms: WaveformIndices) -> Line:
    uid = pulse.id
    w0 = waveforms[(uid, 0)]
    w1 = waveforms[(uid, 1)]
    assert w0[1] == w1[1]
    return Line(
        instruction=Play(wave_0=w0[0], wave_1=w1[0], duration=w0[1]),
        comment=f"id: 0x{uid.hex[:5]}",
    )


def play_duration_swept(registers: dict[ParamRole, Register]) -> list[Instruction]:
    return [
        Play(
            wave_0=registers[ParamRole.PULSE_I],
            wave_1=registers[ParamRole.PULSE_Q],
            duration=4,
        ),
        Wait(duration=registers[ParamRole.DURATION]),
    ]


def play(
    parpulse: ParameterizedPulse,
    waveforms: WaveformIndices,
    acquisitions: dict[MeasureId, AcquisitionSpec],
    merged_vzs: bool,
) -> Block:
    """Process the individual pulse in experiment."""
    pulse = parpulse[0]
    params = parpulse[1]
    if isinstance(pulse, Pulse):
        if merged_vzs:
            assert pulse.relative_phase == 0.0
            return [play_pulse(pulse, waveforms)]
        else:
            phase = int(convert(pulse.relative_phase, Parameter.relative_phase))
            minus_phase = int(convert(-pulse.relative_phase, Parameter.relative_phase))
            duration_sweep = {
                p.role: p.reg for p in params if p.role.value[1] is Parameter.duration
            }
            return (
                (
                    [
                        Add(
                            a=Registers.phase.value,
                            b=phase,
                            destination=Registers.phase.value,
                        )
                    ]
                    if phase != 0
                    else []
                )
                + ([SetPhDelta(value=Registers.phase.value)])
                + (
                    [play_pulse(pulse, waveforms)]
                    if len(duration_sweep) == 0
                    else play_duration_swept(duration_sweep)
                )
                + ([Move(source=minus_phase, destination=Registers.phase.value)])
            )

    if isinstance(pulse, Delay):
        return [
            Line(
                instruction=Wait(duration=int(pulse.duration)),
                comment=f"id: 0x{pulse.id.hex[:5]}",
            )
            if len(params) == 0
            else Wait(duration=next(iter(params)).reg)
        ]
    if isinstance(pulse, VirtualZ):
        if merged_vzs:
            return [
                SetPhDelta(value=int(convert(pulse.phase, Parameter.relative_phase)))
            ]
        else:
            return [
                Add(
                    a=Registers.phase.value,
                    b=int(convert(pulse.phase, Parameter.relative_phase))
                    if len(params) == 0
                    else next(iter(params)).reg,
                    destination=Registers.phase.value,
                )
            ]
    if isinstance(pulse, Acquisition):
        acq = acquisitions[pulse.id]
        return [
            Acquire(
                acquisition=acq.acquisition.index,
                bin=Registers.bin.value,
                duration=acq.duration,
            )
        ]
    if isinstance(pulse, Align):
        raise NotImplementedError("Align operation not yet supported by Qblox.")
    if isinstance(pulse, Readout):
        acq = acquisitions[pulse.id]
        return [
            play_pulse(pulse.probe, waveforms).update(
                {"duration": int(pulse.time_of_flight)}
            ),
            Acquire(
                acquisition=acq.acquisition.index,
                bin=Registers.bin.value,
                duration=int(pulse.acquisition.duration),
            ),
        ]
    raise NotImplementedError(f"Instruction {type(pulse)} unsupported by Qblox driver.")


def event(
    parpulse: ParameterizedPulse,
    waveforms: WaveformIndices,
    acquisitions: dict[MeasureId, AcquisitionSpec],
    merged_vzs: bool,
) -> Block:
    params = parpulse[1]
    return [
        inst
        for block in (
            *(update_instructions(p.role, p.reg) for p in params),
            *(play(parpulse, waveforms, acquisitions, merged_vzs),),
            *(reset_instructions(p.role, p.reg) for p in reversed(list(params))),
        )
        for inst in block
    ]


def experiment(
    sequence: SweepSequence,
    waveforms: WaveformIndices,
    acquisitions: dict[MeasureId, AcquisitionSpec],
    merged_vzs: bool,
) -> Block:
    """Representation of the actual experiment to be executed.

    The parameters' update (`upd_param`) in front of everything is needed to ensure that
    the parameter values for sweepers targeting channels have been updated. The updates
    for those targeting pulses will be triggered by the `play` instruction (and they
    *have to* be local).

    The synchronization (`wait_sync`) will guarantee the common start of all involved
    channels, which is otherwise hard to control (and debug).
    """
    return [UpdParam(duration=4), WaitSync(duration=4)] + [
        inst
        for block in (
            event(pulse, waveforms, acquisitions, merged_vzs) for pulse in sequence
        )
        for inst in block
    ]
