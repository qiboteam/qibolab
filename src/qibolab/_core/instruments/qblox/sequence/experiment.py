from typing import Optional

import numpy as np

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
    Block,
    Instruction,
    Line,
    Play,
    Register,
    SetPhDelta,
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


PHASE_FACTOR = 1e9 / (2 * np.pi)


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
    time_of_flight: Optional[float],
) -> Block:
    """Process the individual pulse in experiment."""
    pulse = parpulse[0]
    params = parpulse[1]

    if isinstance(pulse, Pulse):
        phase = int(convert(pulse.relative_phase, Parameter.relative_phase))
        duration_sweep = {
            p.role: p.reg for p in params if p.role.value[1] is Parameter.duration
        }
        return (
            ([SetPhDelta(value=phase)] if phase != 0 else [])
            + (
                [play_pulse(pulse, waveforms)]
                if len(duration_sweep) == 0
                else play_duration_swept(duration_sweep)
            )
            + ([SetPhDelta(value=-phase)] if phase != 0 else [])
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
        return [
            SetPhDelta(
                value=int(pulse.phase * PHASE_FACTOR)
                if len(params) == 0
                else next(iter(params)).reg
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
        delay = int(time_of_flight) if time_of_flight is not None else 4
        return [
            play_pulse(pulse.probe, waveforms).update({"duration": delay}),
            Acquire(
                acquisition=acq.acquisition.index,
                bin=Registers.bin.value,
                duration=int(pulse.duration),
            ),
        ]
    raise NotImplementedError(f"Instruction {type(pulse)} unsupported by Qblox driver.")


def event(
    parpulse: ParameterizedPulse,
    waveforms: WaveformIndices,
    acquisitions: dict[MeasureId, AcquisitionSpec],
    time_of_flight: Optional[float],
) -> Block:
    params = parpulse[1]
    return [
        inst
        for block in (
            *(update_instructions(p.role, p.reg) for p in params),
            *(play(parpulse, waveforms, acquisitions, time_of_flight),),
            *(reset_instructions(p.role, p.reg) for p in reversed(list(params))),
        )
        for inst in block
    ]


def experiment(
    sequence: SweepSequence,
    waveforms: WaveformIndices,
    acquisitions: dict[MeasureId, AcquisitionSpec],
    time_of_flight: Optional[float],
) -> Block:
    """Representation of the actual experiment to be executed."""
    return [WaitSync(duration=4)] + [
        inst
        for block in (
            event(pulse, waveforms, acquisitions, time_of_flight) for pulse in sequence
        )
        for inst in block
    ]
