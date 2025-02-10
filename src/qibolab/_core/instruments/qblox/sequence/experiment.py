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
    Instruction,
    Play,
    Register,
    SetPhDelta,
    Wait,
)
from .acquisition import AcquisitionSpec, MeasureId
from .asm import Registers
from .sweepers import (
    Param,
    ParameterizedPulse,
    SweepSequence,
    reset_instructions,
    update_instructions,
)
from .waveforms import WaveformIndices, pulse_uid

__all__ = []


PHASE_FACTOR = 1e9 / (2 * np.pi)


def play_pulse(pulse: Pulse, waveforms: WaveformIndices) -> Instruction:
    uid = pulse_uid(pulse)
    w0 = waveforms[(uid, 0)]
    w1 = waveforms[(uid, 1)]
    assert w0[1] == w1[1]
    return Play(wave_0=w0[0], wave_1=w1[0], duration=w0[1])


def play_duration_swept(pulse: Pulse, param: Param) -> Instruction:
    return Play(
        wave_0=param.reg,
        wave_1=Register(number=param.reg.number + 1),
        duration=0,
    )


def play(
    parpulse: ParameterizedPulse,
    waveforms: WaveformIndices,
    acquisitions: dict[MeasureId, AcquisitionSpec],
) -> list[Instruction]:
    """Process the individual pulse in experiment."""
    pulse = parpulse[0]
    param = parpulse[1]

    if isinstance(pulse, Pulse):
        return [
            play_pulse(pulse, waveforms)
            if param is None or param.kind is not Parameter.duration
            else play_duration_swept(pulse, param)
        ]
    if isinstance(pulse, Delay):
        return [Wait(duration=int(pulse.duration) if param is None else param.reg)]
    if isinstance(pulse, VirtualZ):
        return [
            SetPhDelta(
                value=int(pulse.phase * PHASE_FACTOR) if param is None else param.reg
            )
        ]
    if isinstance(pulse, Acquisition):
        acq = acquisitions[str(pulse.id)]
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
        acq = acquisitions[str(pulse.id)]
        return [
            Acquire(
                acquisition=acq.acquisition.index,
                bin=Registers.bin.value,
                duration=4,
            ),
            play_pulse(pulse.probe, waveforms).model_copy(
                update={"duration": int(pulse.duration) - 4}
            ),
        ]
    raise NotImplementedError(f"Instruction {type(pulse)} unsupported by Qblox driver.")


def event(
    parpulse: ParameterizedPulse,
    waveforms: WaveformIndices,
    acquisitions: dict[MeasureId, AcquisitionSpec],
) -> list[Instruction]:
    param = parpulse[1]
    return (
        (update_instructions(param.kind, param.reg) if param is not None else [])
        + play(parpulse, waveforms, acquisitions)
        + (reset_instructions(param.kind, param.reg) if param is not None else [])
    )


def experiment(
    sequence: SweepSequence,
    waveforms: WaveformIndices,
    acquisitions: dict[MeasureId, AcquisitionSpec],
) -> list[Instruction]:
    """Representation of the actual experiment to be executed."""
    return [
        i_
        for block in (event(pulse, waveforms, acquisitions) for pulse in sequence)
        for i_ in block
    ]
