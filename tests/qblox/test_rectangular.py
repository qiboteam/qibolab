"""Unit tests for optimized rectangular pulse implementation using set_awg_offs."""

from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.instruments.qblox.q1asm.ast_ import (
    Add,
    Line,
    Play,
    SetAwgGain,
    SetAwgOffs,
    SetPhDelta,
    Wait,
)
from qibolab._core.instruments.qblox.sequence import compile
from qibolab._core.instruments.qblox.sequence.asm import Registers
from qibolab._core.pulses import Gaussian, Pulse, Rectangular
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper


def test_static_rectangular_pulse():
    pulse = Pulse(duration=100, amplitude=0.5, envelope=Rectangular())
    seq = PulseSequence([("ch1", pulse)])
    options = ExecutionParameters(nshots=1, relaxation_time=1000)
    result = compile(seq, [], options, sampling_rate=1.0, merged_vzs=True)
    q1 = result["ch1"]

    # Waveform memory should be empty
    assert len(q1.waveforms) == 0

    instrs = [
        line.instruction
        for line in q1.program.elements
        if isinstance(line, Line) and line.instruction is not None
    ]

    # Should not contain Play
    assert not any(isinstance(ins, Play) for ins in instrs)

    # Should contain SetAwgOffs and UpdParam
    offs = [ins for ins in instrs if isinstance(ins, SetAwgOffs)]
    assert len(offs) == 2
    assert offs[0].value_0 == 16383
    assert offs[0].value_1 == 0
    assert offs[1].value_0 == 0
    assert offs[1].value_1 == 0

    # Wait duration in the middle is duration - 4 = 96
    waits = [ins for ins in instrs if isinstance(ins, Wait)]
    assert any(w.duration == 96 for w in waits)


def test_static_rectangular_pulse_min_duration():
    # Exactly 4 ns duration
    pulse = Pulse(duration=4, amplitude=0.5, envelope=Rectangular())
    seq = PulseSequence([("ch1", pulse)])
    options = ExecutionParameters(nshots=1, relaxation_time=1000)
    result = compile(seq, [], options, sampling_rate=1.0, merged_vzs=True)
    q1 = result["ch1"]

    assert len(q1.waveforms) == 0

    instrs = [
        line.instruction
        for line in q1.program.elements
        if isinstance(line, Line) and line.instruction is not None
    ]

    offs = [ins for ins in instrs if isinstance(ins, SetAwgOffs)]
    assert len(offs) == 2
    assert offs[0].value_0 == 16383
    assert offs[0].value_1 == 0
    assert offs[1].value_0 == 0
    assert offs[1].value_1 == 0

    # For duration = 4, no intermediate wait is needed
    waits = [ins for ins in instrs if isinstance(ins, Wait)]
    assert not any(w.duration == 0 for w in waits)


def test_long_static_rectangular_pulse():
    # 100_000 ns pulse: exceeds MAX_WAIT (65535) and must be decomposed by transpile
    pulse = Pulse(duration=100_000, amplitude=0.5, envelope=Rectangular())
    seq = PulseSequence([("ch1", pulse)])
    options = ExecutionParameters(nshots=1, relaxation_time=1000)
    result = compile(seq, [], options, sampling_rate=1.0, merged_vzs=True)
    q1 = result["ch1"]

    assert len(q1.waveforms) == 0

    instrs = [
        line.instruction
        for line in q1.program.elements
        if isinstance(line, Line) and line.instruction is not None
    ]
    waits = [ins for ins in instrs if isinstance(ins, Wait)]
    # Decomposed intermediate wait: 100_000 - 4 = 99996 = 34461 + 65535
    assert any(w.duration == 34461 for w in waits)
    assert any(w.duration == 65535 for w in waits)


def test_short_rectangular_pulse_fallback():
    # Duration < 4 cannot be generated with set_awg_offs and should fallback to waveforms
    pulse = Pulse(duration=2, amplitude=0.5, envelope=Rectangular())
    seq = PulseSequence([("ch1", pulse)])
    options = ExecutionParameters(nshots=1, relaxation_time=1000)
    result = compile(seq, [], options, sampling_rate=1.0, merged_vzs=True)
    q1 = result["ch1"]

    assert len(q1.waveforms) == 2
    instrs = [
        line.instruction
        for line in q1.program.elements
        if isinstance(line, Line) and line.instruction is not None
    ]
    assert any(isinstance(ins, Play) and ins.duration == 2 for ins in instrs)


def test_duration_swept_rectangular_pulse():
    pulse = Pulse(duration=100, amplitude=0.5, envelope=Rectangular())
    seq = PulseSequence([("ch1", pulse)])
    sweeper = Sweeper(parameter=Parameter.duration, range=(10, 50, 10), pulses=[pulse])
    options = ExecutionParameters(nshots=1, relaxation_time=1000)
    result = compile(seq, [[sweeper]], options, sampling_rate=1.0, merged_vzs=True)
    q1 = result["ch1"]

    assert len(q1.waveforms) == 0

    instrs = [
        line.instruction
        for line in q1.program.elements
        if isinstance(line, Line) and line.instruction is not None
    ]
    # No Play instruction
    assert not any(isinstance(ins, Play) for ins in instrs)

    # Dynamic wait on register
    reg_waits = [
        ins
        for ins in instrs
        if isinstance(ins, Wait) and not isinstance(ins.duration, int)
    ]
    assert len(reg_waits) == 1

    # Shift sweeper step should be 10
    adds = [
        ins
        for ins in instrs
        if isinstance(ins, Add)
        and ins.b == 10
        and ins.destination == reg_waits[0].duration
    ]
    assert len(adds) >= 1


def test_amplitude_swept_rectangular_pulse():
    pulse = Pulse(duration=100, amplitude=0.5, envelope=Rectangular())
    seq = PulseSequence([("ch1", pulse)])
    sweeper = Sweeper(
        parameter=Parameter.amplitude, range=(0.1, 0.5, 0.1), pulses=[pulse]
    )
    options = ExecutionParameters(nshots=1, relaxation_time=1000)
    result = compile(seq, [[sweeper]], options, sampling_rate=1.0, merged_vzs=True)
    q1 = result["ch1"]

    assert len(q1.waveforms) == 0

    instrs = [
        line.instruction
        for line in q1.program.elements
        if isinstance(line, Line) and line.instruction is not None
    ]

    # No set_awg_gain instruction should be emitted
    assert not any(isinstance(ins, SetAwgGain) for ins in instrs)

    # SetAwgOffs should use the amplitude register and the zero register
    offs = [ins for ins in instrs if isinstance(ins, SetAwgOffs)]
    assert len(offs) == 2
    assert offs[0].value_1 == Registers.zero.value
    assert offs[1].value_0 == 0
    assert offs[1].value_1 == 0


def test_2d_swept_rectangular_pulse():
    pulse = Pulse(duration=100, amplitude=0.5, envelope=Rectangular())
    seq = PulseSequence([("ch1", pulse)])
    sw_amp = Sweeper(
        parameter=Parameter.amplitude, range=(0.1, 0.5, 0.1), pulses=[pulse]
    )
    sw_dur = Sweeper(parameter=Parameter.duration, range=(10, 50, 10), pulses=[pulse])
    options = ExecutionParameters(nshots=1, relaxation_time=1000)
    result = compile(
        seq, [[sw_amp], [sw_dur]], options, sampling_rate=1.0, merged_vzs=True
    )
    q1 = result["ch1"]

    assert len(q1.waveforms) == 0
    instrs = [
        line.instruction
        for line in q1.program.elements
        if isinstance(line, Line) and line.instruction is not None
    ]
    assert not any(isinstance(ins, SetAwgGain) for ins in instrs)
    assert not any(isinstance(ins, Play) for ins in instrs)


def test_rectangular_pulse_with_relative_phase():
    pulse = Pulse(
        duration=100, amplitude=0.5, relative_phase=0.5, envelope=Rectangular()
    )
    seq = PulseSequence([("ch1", pulse)])
    options = ExecutionParameters(nshots=1, relaxation_time=1000)
    result = compile(seq, [], options, sampling_rate=1.0, merged_vzs=False)
    q1 = result["ch1"]

    assert len(q1.waveforms) == 0
    instrs = [
        line.instruction
        for line in q1.program.elements
        if isinstance(line, Line) and line.instruction is not None
    ]
    assert any(isinstance(ins, SetPhDelta) for ins in instrs)


def test_gaussian_pulse_unaffected():
    # Gaussian pulses must continue to use waveforms and set_awg_gain
    pulse = Pulse(duration=40, amplitude=0.5, envelope=Gaussian(rel_sigma=0.2))
    seq = PulseSequence([("ch1", pulse)])
    sweeper = Sweeper(
        parameter=Parameter.amplitude, range=(0.1, 0.5, 0.1), pulses=[pulse]
    )
    options = ExecutionParameters(nshots=1, relaxation_time=1000)
    result = compile(seq, [[sweeper]], options, sampling_rate=1.0, merged_vzs=True)
    q1 = result["ch1"]

    assert len(q1.waveforms) == 2
    instrs = [
        line.instruction
        for line in q1.program.elements
        if isinstance(line, Line) and line.instruction is not None
    ]
    assert any(isinstance(ins, Play) for ins in instrs)
    assert any(isinstance(ins, SetAwgGain) for ins in instrs)
