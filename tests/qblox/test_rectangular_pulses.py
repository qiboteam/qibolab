"""Rectangular pulses with durations of at least 8 ns are synthesized with
`set_awg_offs` instead of waveforms."""

import numpy as np
import pytest

from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.instruments.qblox.q1asm.ast_ import (
    Instr,
    Play,
    Register,
    SetAwgGain,
    SetAwgOffs,
    UpdParam,
    Wait,
)
from qibolab._core.instruments.qblox.sequence.sequence import Q1Sequence, compile
from qibolab._core.instruments.qblox.validate import WAVEFORM_MEMORY, validate_sequence
from qibolab._core.pulses import (
    Acquisition,
    Gaussian,
    Pulse,
    Readout,
    Rectangular,
)
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


def _instructions(q1seq: Q1Sequence) -> list[Instr]:
    """Extract the list of instructions from a compiled Q1ASM sequence."""
    return [line.instruction for line in q1seq.program.lines]


# pulses that are played from waveform memory rather than synthesized through offsets: a
# non-rectangular envelope, and a rectangular one below the 8 ns threshold
NON_OFFSET_PULSE_EXAMPLES = [
    Pulse(duration=40, amplitude=0.5, envelope=Gaussian(rel_sigma=0.2)),
    Pulse(duration=4, amplitude=0.5, envelope=Rectangular()),
]


def test_offset_sweeper_conflicts_with_offset_rectangular_pulse():
    """Sweeping the offset of a channel carrying an offset-based rectangular pulse is
    rejected."""
    pulse = Pulse(duration=40, amplitude=0.5, envelope=Rectangular())
    sweeper = Sweeper(
        parameter=Parameter.offset,
        values=np.array([0.1, 0.2]),
        channels=["ch1"],
    )

    with pytest.raises(ValueError, match="Cannot sweep the offset of channel 'ch1'"):
        _compile([("ch1", pulse)], [[sweeper]])


@pytest.mark.parametrize("pulse", NON_OFFSET_PULSE_EXAMPLES)
def test_non_offset_pulse_is_played_from_waveform_memory(pulse):
    """Pulses that can't be offset-synthesized are played from waveform memory."""
    q1seq = _compile([("ch1", pulse)])["ch1"]
    instrs = _instructions(q1seq)
    assert any(isinstance(i, Play) for i in instrs)
    assert not any(isinstance(i, SetAwgOffs) for i in instrs)
    assert len(q1seq.waveforms) > 0


def test_long_rectangular_pulse_fits_waveform_memory():
    """A very long rectangular pulse uploads no samples, so it cannot overflow waveform
    memory."""
    pulse = Pulse(duration=200_000, amplitude=0.5, envelope=Rectangular())
    # the pulse alone would exceed the waveform memory if it were played
    assert pulse.duration > WAVEFORM_MEMORY
    result = _compile([("ch1", pulse)])
    q1seq = result["ch1"]
    assert len(q1seq.waveforms) == 0
    validate_sequence(q1seq, is_qrm=False)


def test_rectangular_amplitude_sweeper():
    """A swept amplitude feeds the offset directly, without touching the AWG gain."""
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
    assert any(isinstance(i, SetAwgOffs) for i in instrs)


def test_rectangular_pulse_with_relative_phase():
    """A relative phase does not prevent offset synthesis when VZs are not merged."""
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


def test_offset_rectangular_emits_ordered_block():
    """The order of the block is what realizes the pulse, so it is pinned exactly."""
    pulse = Pulse(duration=40, amplitude=0.5, envelope=Rectangular())
    instrs = _instructions(_compile([("ch1", pulse)])["ch1"])
    expected = [
        SetAwgOffs(value_0=16383, value_1=0),
        UpdParam(duration=4),
        Wait(duration=36),
        SetAwgOffs(value_0=0, value_1=0),
        UpdParam(duration=4),
    ]
    start = next(i for i, instr in enumerate(instrs) if isinstance(instr, SetAwgOffs))
    assert instrs[start : start + len(expected)] == expected


@pytest.mark.parametrize("pulse", NON_OFFSET_PULSE_EXAMPLES)
def test_offset_sweeper_allowed_when_pulse_is_played(pulse):
    """Only rectangular pulses can be synthesized through offsets.

    Any pulse that is played from waveform memory (rather than synthesized through
    offsets) may still be swept on its offset.
    """
    sweeper = Sweeper(
        parameter=Parameter.offset,
        values=np.array([0.1, 0.2]),
        channels=["ch1"],
    )
    instrs = _instructions(_compile([("ch1", pulse)], [[sweeper]])["ch1"])
    assert any(isinstance(i, Play) for i in instrs)
    # the offset sweeper still drives `set_awg_offs` through a register
    assert any(
        isinstance(i, SetAwgOffs) and isinstance(i.value_0, Register) for i in instrs
    )


def test_readout_probe_is_not_synthesized_through_offsets():
    """Known limitation: readout probe pulses are not synthesized through offsets."""
    probe = Pulse(duration=1000, amplitude=0.5, envelope=Rectangular())
    readout = Readout(probe=probe, acquisition=Acquisition(duration=1000))
    q1seq = _compile([("ch1", readout)])["ch1"]
    assert len(q1seq.waveforms) == 2
    assert any(isinstance(i, Play) for i in _instructions(q1seq))
