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
    Lineable,
    Move,
    Play,
    Register,
    SetAwgOffs,
    SetPhDelta,
    UpdParam,
    Wait,
    WaitSync,
)
from .acquisition import AcquisitionSpec, MeasureId
from .asm import Registers, convert
from .sweepers import (
    Param,
    ParameterizedPulse,
    ParamRole,
    SweepSequence,
    reset_instructions,
    update_instructions,
)
from .waveforms import PulseRealization, WaveformIndices

__all__ = []


def _play_waveforms(pulse: Pulse, waveforms: WaveformIndices) -> Line:
    uid = pulse.id
    w0 = waveforms[(uid, 0)]
    w1 = waveforms[(uid, 1)]
    assert w0[1] == w1[1]
    return Line(
        instruction=Play(wave_0=w0[0], wave_1=w1[0], duration=w0[1]),
        comment=f"id: 0x{uid.hex[:5]}",
    )


def _play_duration_swept(registers: dict[ParamRole, Register]) -> list[Instruction]:
    return [
        Play(
            wave_0=registers[ParamRole.PULSE_I],
            wave_1=registers[ParamRole.PULSE_Q],
            duration=4,
        ),
        Wait(duration=registers[ParamRole.DURATION]),
    ]


def _play_pulse(
    pulse: Pulse,
    waveforms: WaveformIndices,
    duration_sweep: dict[ParamRole, Register],
) -> list[Instruction] | list[Line]:
    return (
        [_play_waveforms(pulse, waveforms)]
        if len(duration_sweep) == 0
        else _play_duration_swept(duration_sweep)
    )


def _process_rectangular(pulse: Pulse, params: set[Param]) -> list[Lineable]:
    """Emit Q1ASM for a rectangular pulse using `set_awg_offs`.

    The constant level is set as an AWG offset (the NCO is still oscillating and
    phase rotations on top of it work as for played waveforms), so no sample has
    to be stored in the waveform memory::

        set_awg_offs <amp>, 0
        upd_param    4
        wait         <duration - 4>
        set_awg_offs 0, 0
        upd_param    4

    The trailing `upd_param` is needed because `wait` does not latch parameters,
    so the offset reset has to be explicitly applied at the pulse end.
    """

    duration_sweep = {p.role: p.reg for p in params if p.role is ParamRole.DURATION}
    amplitude_sweep = {p.role: p.reg for p in params if p.role is ParamRole.AMPLITUDE}

    # The rectangular pulse is played only on path-0 (the I-channel). The value of the
    # Q-channel signal is always 0.0.
    #
    # `set_awg_offs` in Qblox expects both arguments of the same type (register or
    # immediate), therefore `zero` is assigned to match `amplitude`. See:
    # https://docs.qblox.com/en/main/products/qblox_instruments/q1/index.html#set_awg_offs-instt
    if amplitude_sweep:
        # If the amplitude is swept, then pulse.amplitude is just a placeholder. The
        # intended value is defined by the sweepers' range, and a register is then used
        # in place of an immediate.
        amplitude = amplitude_sweep[ParamRole.AMPLITUDE]
        zero = Registers.zero.value
    else:
        # If the amplitude is fixed, convert the normalized amplitude assigned to the
        # pulse to units of set_awg_offs.
        amplitude = int(convert(pulse.amplitude, Parameter.amplitude))
        zero = 0

    # The first `upd_param` below takes 4 ns so these don't have to be in the wait.
    # For swept durations the register values are already shifted by those 4 ns
    # (see `sweepers._duration_shift`).
    wait_duration = (
        duration_sweep[ParamRole.DURATION]
        if duration_sweep
        else int(pulse.duration) - 4
    )

    # NOTE: The UpdParam below causes the realtime duration of the pulse to be 4 ns
    # longer than the requested duration.
    return [
        SetAwgOffs(value_0=amplitude, value_1=zero),
        Line(
            instruction=UpdParam(duration=4),
            comment=f"id: 0x{pulse.id.hex[:5]}",
        ),
        Wait(duration=wait_duration),
        SetAwgOffs(value_0=0, value_1=0),
        UpdParam(duration=4),
    ]


def _process_pulse(
    pulse: Pulse,
    params: set[Param],
    pulse_realization: PulseRealization,
    merged_vzs: bool,
):
    """
    If merged_vzs is True, all virtual-Z gates are merged and phase handling is done in
    _process_virtualz.

    If merged_vzs is False, a nonzero pulse.relative_phase is added to previously
    accumulated phase deltas in Registers.phase_delta and must be applied with
    SetPhDelta before playing the pulse.
    """
    duration_sweep = {
        p.role: p.reg for p in params if p.role.value[1] is Parameter.duration
    }
    # Rectangular pulses with duration >= 8 ns are implemented using `set_awg_offs`. For
    # all other pulses, waveforms are played.
    pulse_instructions = (
        _process_rectangular(pulse, params)
        if pulse.id in pulse_realization.offset_pulses
        else _play_pulse(pulse, pulse_realization.waveform_indices, duration_sweep)
    )
    if merged_vzs:
        assert pulse.relative_phase == 0.0
        return pulse_instructions
    else:
        phase = int(convert(pulse.relative_phase, Parameter.relative_phase))
        minus_phase = int(convert(-pulse.relative_phase, Parameter.relative_phase))
        return (
            (
                [
                    Add(
                        a=Registers.phase_delta.value,
                        b=phase,
                        destination=Registers.phase_delta.value,
                    )
                ]
                if phase != 0
                else []
            )
            + ([SetPhDelta(value=Registers.phase_delta.value)])
            + pulse_instructions
            + ([Move(source=minus_phase, destination=Registers.phase_delta.value)])
        )


def _process_delay(pulse: Delay, params: set[Param]):
    if len(params) == 0:
        return [
            Line(
                instruction=Wait(duration=int(pulse.duration)),
                comment=f"id: 0x{pulse.id.hex[:5]}",
            )
        ]
    else:
        return [Wait(duration=next(iter(params)).reg)]


def _process_virtualz(pulse: VirtualZ, params: set[Param], merged_vzs: bool):
    """
    If merged_vzs is True, there is only a single VirtualZ between plays, so it apply
    the phase directly using SetPhDelta.

    If merged_vzs is False, accumulate the phase delta in Registers.phase_delta. If
    params are provided, take the value from the corresponding register.
    """
    if merged_vzs:
        return [SetPhDelta(value=int(convert(pulse.phase, Parameter.phase)))]
    else:
        return [
            Add(
                a=Registers.phase_delta.value,
                b=int(convert(pulse.phase, Parameter.relative_phase))
                if len(params) == 0
                else next(iter(params)).reg,
                destination=Registers.phase_delta.value,
            )
        ]


def _process_acquisition(
    pulse: Acquisition, acquisitions: dict[MeasureId, AcquisitionSpec]
):
    acq = acquisitions[pulse.id]
    return [
        Acquire(
            acquisition=acq.acquisition.index,
            bin=Registers.bin.value,
            duration=acq.duration,
        )
    ]


def _process_readout(
    pulse: Readout,
    waveforms: WaveformIndices,
    acquisitions: dict[MeasureId, AcquisitionSpec],
):
    acq = acquisitions[pulse.id]
    return [
        _play_waveforms(pulse.probe, waveforms).update(
            {"duration": int(pulse.time_of_flight)}
        ),
        Acquire(
            acquisition=acq.acquisition.index,
            bin=Registers.bin.value,
            duration=int(pulse.acquisition.duration),
        ),
    ]


def play(
    parpulse: ParameterizedPulse,
    pulse_realization: PulseRealization,
    acquisitions: dict[MeasureId, AcquisitionSpec],
    merged_vzs: bool,
) -> Block:
    """Process the individual pulse in experiment."""
    pulse = parpulse[0]
    params = parpulse[1]
    if isinstance(pulse, Pulse):
        return _process_pulse(pulse, params, pulse_realization, merged_vzs)
    if isinstance(pulse, Delay):
        return _process_delay(pulse, params)
    if isinstance(pulse, VirtualZ):
        return _process_virtualz(pulse, params, merged_vzs)
    if isinstance(pulse, Acquisition):
        return _process_acquisition(pulse, acquisitions)
    if isinstance(pulse, Align):
        raise NotImplementedError("Align operation not yet supported by Qblox.")
    if isinstance(pulse, Readout):
        return _process_readout(pulse, pulse_realization.waveform_indices, acquisitions)
    raise NotImplementedError(f"Instruction {type(pulse)} unsupported by Qblox driver.")


def event(
    parpulse: ParameterizedPulse,
    pulse_realization: PulseRealization,
    acquisitions: dict[MeasureId, AcquisitionSpec],
    merged_vzs: bool,
) -> Block:
    pulse, params = parpulse
    # For offset rectangular pulses the amplitude sweeper works differently: if a pulse
    # is implemented through a waveform, the implementation varies the waveform gain
    # with `set_awg_gain` (output = waveform * gain + offset), which cannot change the
    # amplitude set through `set_awg_offs`. Instead, the swept amplitude is written
    # straight into the pulse's own `set_awg_offs` in `_process_rectangular` and the
    # AMPLITUDE parameter is excluded from the usual gain update/reset around the event.
    is_offset = pulse.id in pulse_realization.offset_pulses
    sweep_params = [
        p for p in params if not (is_offset and p.role is ParamRole.AMPLITUDE)
    ]
    return [
        inst
        for block in (
            *(update_instructions(p.role, p.reg) for p in sweep_params),
            *(play(parpulse, pulse_realization, acquisitions, merged_vzs),),
            *(reset_instructions(p.role, p.reg) for p in reversed(sweep_params)),
        )
        for inst in block
    ]


def experiment(
    sequence: SweepSequence,
    pulse_realization: PulseRealization,
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
            event(pulse, pulse_realization, acquisitions, merged_vzs)
            for pulse in sequence
        )
        for inst in block
    ]
