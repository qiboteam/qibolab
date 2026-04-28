import pytest

from qibolab._core.instruments.qblox.q1asm.ast_ import (
    Instruction,
    Line,
    Loop,
    Move,
    Play,
    Program,
    Reference,
    Register,
    UpdParam,
    Wait,
)
from qibolab._core.instruments.qblox.sequence.transpile import transpile


def _instructions(prog: Program) -> list[Instruction]:
    """ "Helper function to convert program to instructions"""
    instructions = [
        element.instruction for element in prog.elements if isinstance(element, Line)
    ]
    return instructions


def test_updparam_subtracts_from_wait():
    block = [Wait(duration=10), UpdParam(duration=4)]
    prog = transpile(block)
    instructions = _instructions(prog)
    assert instructions == [Wait(duration=6), UpdParam(duration=4)]


def test_multiple_updparam_accumulation():
    block = [Wait(duration=10), UpdParam(duration=4), UpdParam(duration=2)]
    prog = transpile(block)
    instructions = _instructions(prog)
    assert instructions == [
        Wait(duration=4),
        UpdParam(duration=4),
        UpdParam(duration=2),
    ]


def test_updparam_dont_subtract_from_register_wait():
    # UpdParam should not affect Wait with Register duration
    block = [Wait(duration=5), Wait(duration=Register(number=1)), UpdParam(duration=4)]
    prog = transpile(block)
    instructions = _instructions(prog)
    assert instructions == [
        Wait(duration=1),
        Wait(duration=Register(number=1)),
        UpdParam(duration=4),
    ]
    # Test also for the inverted case where the Wait with Register comes first
    block_inverted = [
        Wait(duration=Register(number=1)),
        Wait(duration=5),
        UpdParam(duration=4),
    ]
    prog_inverted = transpile(block_inverted)
    instructions_inverted = _instructions(prog_inverted)
    assert instructions_inverted == [
        Wait(duration=Register(number=1)),
        Wait(duration=1),
        UpdParam(duration=4),
    ]


def test_negative_duration_assertion():
    # Wait(5), UpdParam(10) should fail because we want to subtract 10ns from the Wait.
    block = [Wait(duration=5), UpdParam(duration=10)]
    with pytest.raises(ValueError):
        transpile(block)


def test_long_wait_decomposition_and_subtraction():
    # Wait(70000), UpdParam(4) should decompose and subtract
    block = [Wait(duration=70000), UpdParam(duration=4)]
    prog = transpile(block)
    instructions = _instructions(prog)
    assert instructions == [
        Wait(duration=4461),
        Move(source=1, destination=Register(number=3)),
        Wait(duration=65535),
        Loop(a=Register(number=3), address=Reference(label="wait1")),
        UpdParam(duration=4),
    ]


def test_interleaved_updparam_wait():
    block = [
        Wait(duration=10),
        UpdParam(duration=3),
        Wait(duration=8),
        UpdParam(duration=2),
    ]
    prog = transpile(block)
    instructions = _instructions(prog)
    assert instructions == [
        Wait(duration=7),
        UpdParam(duration=3),
        Wait(duration=6),
        UpdParam(duration=2),
    ]


def test_updparam_before_play_increments_counter():
    block = [
        Play(wave_0=0, wave_1=0, duration=10),
        UpdParam(duration=4),
        Wait(duration=8),
        UpdParam(duration=2),
    ]
    prog = transpile(block)
    instructions = _instructions(prog)
    assert instructions == [
        Play(wave_0=0, wave_1=0, duration=6),
        UpdParam(duration=4),
        Wait(duration=6),
        UpdParam(duration=2),
    ]


def test_updparam_does_not_subtract_from_play_with_register():
    block_reg0 = [
        Play(wave_0=0, wave_1=0, duration=10),
        Play(wave_0=Register(number=0), wave_1=Register(number=1), duration=10),
        UpdParam(duration=4),
    ]
    prog_reg0 = transpile(block_reg0)
    instructions_reg0 = _instructions(prog_reg0)
    assert instructions_reg0 == [
        Play(wave_0=0, wave_1=0, duration=6),
        Play(wave_0=Register(number=0), wave_1=Register(number=1), duration=10),
        UpdParam(duration=4),
    ]
