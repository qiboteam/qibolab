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
    Add,
    Instruction,
    Play,
    Register,
    SetPhDelta,
    Wait,
)
from .acquisition import AcquisitionSpec, MeasureId
from .asm import Registers, convert
from .sweepers import (
    Param,
    ParameterizedPulse,
    SweepSequence,
    reset_instructions,
    update_instructions,
)
from .waveforms import WaveformIndices

__all__ = []


PHASE_FACTOR = 1e9 / (2 * np.pi)


def play_pulse(pulse: Pulse, waveforms: WaveformIndices) -> Instruction:
    uid = pulse.id
    w0 = waveforms[(uid, 0)]
    w1 = waveforms[(uid, 1)]
    assert w0[1] == w1[1]
    return Play(wave_0=w0[0], wave_1=w1[0], duration=w0[1])


def play_duration_swept(param: Param) -> list[Instruction]:
    qreg = Register(number=param.reg.number + 1)
    return [
        Add(a=param.reg, b=1, destination=qreg),
        Play(
            wave_0=param.reg,
            wave_1=qreg,
            duration=4,
        ),
        Wait(duration=Register(number=param.reg.number + 2)),
    ]


def play(
    parpulse: ParameterizedPulse,
    waveforms: WaveformIndices,
    acquisitions: dict[MeasureId, AcquisitionSpec],
    time_of_flight: Optional[float],
) -> list[Instruction]:
    """Process the individual pulse in experiment."""
    pulse = parpulse[0]
    params = parpulse[1]

    if isinstance(pulse, Pulse):
        # breakpoint()
        phase = int(convert(pulse.relative_phase, Parameter.relative_phase))
        duration_sweep = min(
            (p for p in params if p.role.value is Parameter.duration),
            key=lambda p: p.reg.number,
            default=None,
        )
        return (
            ([SetPhDelta(value=phase)] if phase != 0 else [])
            + (
                [play_pulse(pulse, waveforms)]
                if duration_sweep is None
                else play_duration_swept(duration_sweep)
            )
            + ([SetPhDelta(value=-phase)] if phase != 0 else [])
        )
    if isinstance(pulse, Delay):
        return [
            Wait(
                duration=int(pulse.duration)
                if len(params) == 0
                else next(iter(params)).reg
            )
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
            play_pulse(pulse.probe, waveforms).model_copy(update={"duration": delay}),
            Acquire(
                acquisition=acq.acquisition.index,
                bin=Registers.bin.value,
                duration=int(pulse.duration) - delay,
            ),
        ]
    raise NotImplementedError(f"Instruction {type(pulse)} unsupported by Qblox driver.")


def event(
    parpulse: ParameterizedPulse,
    waveforms: WaveformIndices,
    acquisitions: dict[MeasureId, AcquisitionSpec],
    time_of_flight: Optional[float],
) -> list[Instruction]:
    params = parpulse[1]
    return (
        [inst for p in params for inst in update_instructions(p.role, p.reg)]
        + play(parpulse, waveforms, acquisitions, time_of_flight)
        + [
            inst
            for p in reversed(list(params))
            for inst in reset_instructions(p.role, p.reg)
        ]
    )


def experiment(
    sequence: SweepSequence,
    waveforms: WaveformIndices,
    acquisitions: dict[MeasureId, AcquisitionSpec],
    time_of_flight: Optional[float],
) -> list[Instruction]:
    """Representation of the actual experiment to be executed."""
    return [
        i_
        for block in (
            event(pulse, waveforms, acquisitions, time_of_flight) for pulse in sequence
        )
        for i_ in block
    ]
