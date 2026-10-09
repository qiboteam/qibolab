from qibolab._core.instruments.qblox.q1asm.ast_ import (
    Line,
    Loop,
    Move,
    Reference,
    Register,
    Sub,
    Wait,
)
from qibolab._core.instruments.qblox.sequence.asm import Registers
from qibolab._core.instruments.qblox.sequence.finalize import (
    DEFAULT_PIPELINE,
    move,
    transform,
    wait,
)


def test_merge_wait_consecutive():
    """Consecutive immediate waits are merged into one."""
    block = [
        Wait(duration=4),
        Wait(duration=6),
        Wait(duration=10),
    ]
    result = transform(block, (wait.merge_wait,))
    assert len(result) == 1
    instr = result[0].instruction
    assert isinstance(instr, Wait)
    assert instr.duration == 20


def test_merge_wait_preserves_comment_and_label():
    block = [
        Wait(duration=4),
        Line(instruction=Wait(duration=6), comment="mid wait"),
    ]
    result = transform(block, (wait.merge_wait,))
    assert len(result) == 1
    assert result[0].comment == "mid wait"
    assert result[0].instruction.duration == 10


def test_merge_wait_stops_at_label():
    """A labeled wait terminates the merge; it is kept as its own line."""
    block = [
        Wait(duration=4),
        Line(instruction=Wait(duration=6), label="mid"),
    ]
    result = transform(block, (wait.merge_wait,))
    assert len(result) == 2
    assert result[0].instruction.duration == 4
    assert result[1].label == "mid"
    assert result[1].instruction.duration == 6


def test_merge_wait_splits_at_non_wait():
    block = [
        Wait(duration=4),
        Move(source=1, destination=Registers.shots.value),
        Wait(duration=6),
    ]
    result = transform(block, (wait.merge_wait,))
    assert [type(el.instruction) for el in result] == [Wait, Move, Wait]
    assert result[0].instruction.duration == 4
    assert result[2].instruction.duration == 6


def test_merge_wait_zero_duration_dropped():
    block = [Wait(duration=0)]
    result = transform(block, (wait.merge_wait,))
    assert result == []


def test_long_wait_split():
    """A wait exceeding MAX_WAIT is split into a loop of bounded waits."""
    duration = wait.MAX_WAIT * 3 + 17
    block = [Wait(duration=duration)]
    result = transform(block, (wait.long_wait,))
    assert [type(el.instruction) for el in result] == [Wait, Move, Wait, Loop]

    # remainder, iteration count, bounded wait in loop, loop back
    assert result[0].instruction.duration == 17
    assert result[1].instruction.source == 3
    assert result[2].instruction.duration == wait.MAX_WAIT
    label = result[2].label
    assert result[3].instruction.address == Reference(label=label)


def test_long_wait_short_wait_untouched():
    block = [Wait(duration=100)]
    result = transform(block, (wait.long_wait,))
    assert len(result) == 1
    assert result[0].instruction.duration == 100


def test_long_wait_sequential_labels():
    block = [Wait(duration=wait.MAX_WAIT + 1), Wait(duration=wait.MAX_WAIT + 1)]
    result = transform(block, (wait.long_wait,))
    labels = {el.label for el in result if el.label is not None}
    assert labels == {"wait0", "wait1"}


def test_negative_immediate_move():
    """Negative immediate in a move is compiled to zeroing + subtraction."""
    register = Register(number=7)
    block = [Move(source=-5, destination=register)]
    result = transform(block, (move.negative_immediate,))
    assert [type(el.instruction) for el in result] == [Move, Sub]
    assert result[0].instruction.source == 0
    assert result[0].instruction.destination == register
    assert result[1].instruction.b == 5
    assert result[1].instruction.destination == register


def test_positive_immediate_move_untouched():
    register = Register(number=7)
    block = [Move(source=5, destination=register)]
    result = transform(block, (move.negative_immediate,))
    assert len(result) == 1
    assert result[0].instruction.source == 5


def test_default_pipeline_combined():
    """Default pipeline merges waits and expands long ones."""
    block = [
        Wait(duration=4),
        Wait(duration=wait.MAX_WAIT * 2 + 1),
        Move(source=-3, destination=Registers.shots.value),
        Wait(duration=2),
    ]
    result = transform(block, DEFAULT_PIPELINE)
    waits = [el for el in result if isinstance(el.instruction, Wait)]
    moves = [el for el in result if isinstance(el.instruction, Move)]
    subs = [el for el in result if isinstance(el.instruction, Sub)]
    loops = [el for el in result if isinstance(el.instruction, Loop)]

    # long wait expanded into: remainder wait, loop wait, and the two
    # surrounding short waits (4 and 2) remain as immediate waits
    assert len(loops) == 1
    assert len(subs) == 1
    assert subs[0].instruction.b == 3
    # wait durations: 5 (merged 4 + remainder 1), MAX_WAIT (loop body), 2 (short)
    assert {w.instruction.duration for w in waits} == {
        5,
        wait.MAX_WAIT,
        2,
    }
    # the loop moves the iteration count to the preassigned wait register
    wait_move = [m for m in moves if m.instruction.destination == Registers.wait.value]
    assert len(wait_move) == 1
