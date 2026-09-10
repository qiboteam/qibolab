"""Rectangular pulses are synthesized with `set_awg_offs` instead of waveforms."""

import numpy as np

from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.instruments.qblox.q1asm.ast_ import (
    Move,
    Play,
    SetAwgGain,
    SetAwgOffs,
    UpdParam,
    Wait,
)
from qibolab._core.instruments.qblox.sequence.asm import Registers
from qibolab._core.instruments.qblox.sequence.sequence import compile
from qibolab._core.instruments.qblox.validate import validate_sequence
from qibolab._core.pulses import Gaussian, Pulse, Rectangular
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper


def _compile(sequence, sweepers=None, merged_vzs=True):
    options = ExecutionParameters(nshots=1, relaxation_time=100)
    return compile(
        PulseSequence(sequence),
        sweepers or [],
        options,
        sampling_rate=1.0,
        merged_vzs=merged_vzs,
    )


def _instructions(q1seq):
    return [line.instruction for line in q1seq.program.lines]


def test_static_rectangular_pulse_uses_offsets():
    pulse = Pulse(duration=40, amplitude=0.5, envelope=Rectangular())
    result = _compile([("ch1", pulse)])
    q1seq = result["ch1"]

    # no waveform is uploaded, the level is set through the offset
    assert len(q1seq.waveforms) == 0
    instrs = _instructions(q1seq)
    assert not any(isinstance(i, Play) for i in instrs)
    assert SetAwgOffs(value_0=16383, value_1=0) in instrs
    assert SetAwgOffs(value_0=0, value_1=0) in instrs
    # the pulse is `upd_param 4` plus a wait for the remaining duration
    assert Wait(duration=36) in instrs
    # and the offset reset is latched by a final upd_param
    assert sum(isinstance(i, UpdParam) for i in instrs) == 3


def test_short_rectangular_pulse_falls_back_to_waveforms():
    # below 4 ns the offset start/stop cannot be separated, so it is played back
    pulse = Pulse(duration=2, amplitude=0.5, envelope=Rectangular())
    result = _compile([("ch1", pulse)])
    q1seq = result["ch1"]

    assert len(q1seq.waveforms) == 2
    instrs = _instructions(q1seq)
    assert any(isinstance(i, Play) and i.duration == 2 for i in instrs)
    assert not any(isinstance(i, SetAwgOffs) for i in instrs)


def test_short_duration_swept_rectangular_pulse_falls_back():
    # a duration sweeper containing values below 4 ns falls back to waveforms
    pulse = Pulse(duration=8, amplitude=0.5, envelope=Rectangular())
    sweeper = Sweeper(
        parameter=Parameter.duration,
        range=(2, 10, 2),
        pulses=[pulse],
    )
    result = _compile([("ch1", pulse)], [[sweeper]])
    q1seq = result["ch1"]

    assert len(q1seq.waveforms) > 0
    instrs = _instructions(q1seq)
    assert any(isinstance(i, Play) for i in instrs)
    assert not any(isinstance(i, SetAwgOffs) for i in instrs)


def test_shaped_pulse_still_played():
    pulse = Pulse(duration=40, amplitude=0.5, envelope=Gaussian(rel_sigma=0.2))
    result = _compile([("ch1", pulse)])
    q1seq = result["ch1"]
    instrs = _instructions(q1seq)
    assert any(isinstance(i, Play) for i in instrs)
    assert not any(isinstance(i, SetAwgOffs) for i in instrs)
    assert len(q1seq.waveforms) > 0


def test_long_rectangular_pulse_fits_waveform_memory():
    pulse = Pulse(duration=200_000, amplitude=0.5, envelope=Rectangular())
    result = _compile([("ch1", pulse)])
    q1seq = result["ch1"]
    # no samples at all, hence no waveform-memory issue
    assert len(q1seq.waveforms) == 0
    validate_sequence(q1seq, is_qrm=False)


def test_rectangular_amplitude_sweeper():
    pulse = Pulse(duration=40, amplitude=0.5, envelope=Rectangular())
    sweeper = Sweeper(
        parameter=Parameter.amplitude,
        values=np.array([0.1, 0.2, 0.3]),
        pulses=[pulse],
    )
    result = _compile([("ch1", pulse)], [[sweeper]])
    q1seq = result["ch1"]

    assert len(q1seq.waveforms) == 0
    instrs = _instructions(q1seq)
    # gain is not used, the amplitude register feeds the offset directly
    assert not any(isinstance(i, SetAwgGain) for i in instrs)
    assert any(
        isinstance(i, SetAwgOffs) and i.value_1 == Registers.zero.value for i in instrs
    )


def test_rectangular_duration_sweeper():
    pulse = Pulse(duration=40, amplitude=0.5, envelope=Rectangular())
    values = np.array([20.0, 40.0, 60.0])
    sweeper = Sweeper(
        parameter=Parameter.duration,
        values=values,
        pulses=[pulse],
    )
    result = _compile([("ch1", pulse)], [[sweeper]])
    q1seq = result["ch1"]

    # no per-duration waveform is uploaded
    assert len(q1seq.waveforms) == 0
    instrs = _instructions(q1seq)
    assert not any(isinstance(i, Play) for i in instrs)
    # the duration drives the hold wait directly (already reduced by 4 ns)
    waits = [
        i for i in instrs if isinstance(i, Wait) and not isinstance(i.duration, int)
    ]
    assert len(waits) == 1
    moves = {
        line.instruction.source
        for line in q1seq.program.lines
        if isinstance(line.instruction, Move)
        and line.comment is not None
        and "duration.DURATION" in line.comment
    }
    assert moves == {int(values[0]) - 4}


def test_rectangular_pulse_with_relative_phase():
    pulse = Pulse(
        duration=40,
        amplitude=0.5,
        envelope=Rectangular(),
        relative_phase=0.3,
    )
    result = _compile([("ch1", pulse)], merged_vzs=False)
    instrs = _instructions(result["ch1"])
    assert SetAwgOffs(value_0=16383, value_1=0) in instrs
    assert not any(isinstance(i, Play) for i in instrs)
