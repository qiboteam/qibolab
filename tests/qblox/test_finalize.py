import pytest

from qibolab._core.instruments.qblox.q1asm.ast_ import (
    Add,
    And,
    Arithmetic,
    Asl,
    Asr,
    Block,
    Line,
    Loop,
    Move,
    Nop,
    Not,
    Or,
    Reference,
    Register,
    SetPhDelta,
    Stop,
    Sub,
    Wait,
    Xor,
)
from qibolab._core.instruments.qblox.sequence.asm import Registers
from qibolab._core.instruments.qblox.sequence.finalize import (
    DEFAULT_PIPELINE,
    transform,
    traverse,
)
from qibolab._core.instruments.qblox.sequence.finalize.components import (
    BlockRule,
    LineRule,
)
from qibolab._core.instruments.qblox.sequence.finalize.rules import move, register, wait


@pytest.mark.parametrize(
    "instruction",
    [
        Move(source=5, destination="R7"),
        Move(source="R6", destination="R7"),
        Not(source="R6", destination="R7"),
        Add(a="R6", b=5, destination="R7"),
        Sub(a="R6", b=5, destination="R7"),
        And(a="R6", b=5, destination="R7"),
        Or(a="R6", b=5, destination="R7"),
        Xor(a="R6", b=5, destination="R7"),
        Asl(a="R6", b=5, destination="R7"),
        Asr(a="R6", b=5, destination="R7"),
    ],
)
def test_update_nop_after_arithmetic(instruction):
    line = Line(instruction=instruction, label="update", comment="register update")
    result = transform([line], ((register.update_nop,),))
    assert result == [line, Line.instr(Nop())]


def test_update_nop_leaves_other_instructions_untouched():
    block = [
        SetPhDelta(value="R7"),
        Wait(duration=4),
        Loop(a="R7", address="@start"),
        Nop(),
        Stop(),
    ]
    result = transform(block, ((register.update_nop,),))
    assert [line.instruction for line in result] == block


def test_update_nop_between_consecutive_updates():
    block = [
        Move(source=0, destination="R7"),
        Add(a="R7", b=5, destination="R7"),
        Sub(a="R7", b=3, destination="R7"),
        SetPhDelta(value="R7"),
    ]
    result = transform(block, ((register.update_nop,),))
    assert [line.instruction for line in result] == [
        block[0],
        Nop(),
        block[1],
        Nop(),
        block[2],
        Nop(),
        block[3],
    ]


def test_default_pipeline_negative_move_nops():
    line = Line(
        instruction=Move(source=-5, destination="R7"),
        label="negative",
        comment="initialize register",
    )
    result = transform([line], DEFAULT_PIPELINE)
    assert [line.instruction for line in result] == [
        Move(source=0, destination="R7"),
        Nop(),
        Sub(a="R7", b=5, destination="R7"),
        Nop(),
    ]
    assert result[0].label == line.label
    assert result[0].comment == line.comment
    assert all(line.label is None for line in result[1:])


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
    for index, line in enumerate(result):
        if isinstance(line.instruction, Arithmetic):
            assert isinstance(result[index + 1].instruction, Nop)
    assert sum(isinstance(line.instruction, Nop) for line in result) == 3


@pytest.mark.parametrize("step", [wait.merge_wait, wait.long_wait, (wait.long_wait,)])
def test_traverse_returns_flat_lines(step):
    result = traverse([Wait(duration=4), Wait(duration=6)], step)
    assert all(isinstance(line, Line) for line in result)
    assert traverse([], step) == []


def test_empty_pipeline_normalizes_lines():
    annotated = Line(instruction=Wait(duration=6), label="keep", comment="keep")
    assert transform([Wait(duration=4), annotated], ()) == [
        Line.instr(Wait(duration=4)),
        annotated,
    ]
    assert transform([], DEFAULT_PIPELINE) == []


def test_line_rule_carries_match_and_map_state():
    seen = []

    def match(line: Line, state: int) -> tuple[bool, int]:
        seen.append(state)
        return state > 0, state + 1

    def map_(line: Line, state: int) -> tuple[Block, int]:
        return [Wait(duration=state)], state + 10

    rule = LineRule(init_state=lambda: 0, match=match, map=map_)
    block = [
        Wait(duration=4),
        Line(instruction=Wait(duration=6), label="target", comment="annotation"),
        Wait(duration=8),
    ]
    result = traverse(block, rule)
    assert seen == [0, 1, 12]
    assert result == [
        Line.instr(Wait(duration=4)),
        Line(instruction=Wait(duration=2), label="target", comment="annotation"),
        Line.instr(Wait(duration=13)),
    ]


def test_line_alternatives_retain_skipped_rule_state():
    seen = []

    def first_match(line: Line, state: None) -> tuple[bool, None]:
        assert isinstance(line.instruction, Wait)
        return line.instruction.duration == 6, state

    def second_match(line: Line, state: int) -> tuple[bool, int]:
        seen.append(state)
        return False, state + 1

    first = LineRule(
        init_state=lambda: None,
        match=first_match,
        map=lambda line, state: ([line], state),
    )
    second = LineRule(
        init_state=lambda: 0,
        match=second_match,
        map=lambda line, state: ([line], state),
    )
    result = traverse([Wait(duration=n) for n in [4, 6, 8]], (first, second))
    assert seen == [0, 1]
    assert [line.instruction.duration for line in result] == [4, 6, 8]


@pytest.mark.parametrize("restart", [False, True])
def test_block_rule_retains_start_state_at_boundary(restart):
    events = []

    def start(line: Line, state: int) -> tuple[bool, int]:
        events.append(("start", state))
        return state == 0 or restart, state + 1

    def end(line: Line, state: int) -> tuple[bool, int]:
        events.append(("end", state))
        return True, state + 10

    def map_(lines: list[Line], state: int) -> tuple[Block, int]:
        events.append(("map", state))
        return [Wait(duration=state)], state + 100

    rule = BlockRule(init_state=lambda: 0, start=start, end=end, map=map_)
    result = traverse([Wait(duration=4), Wait(duration=6)], rule)
    assert events == [
        ("start", 0),
        ("end", 1),
        ("start", 11),
        ("map", 12),
        *([("map", 112)] if restart else []),
    ]
    assert result == [
        Line.instr(Wait(duration=12)),
        Line.instr(Wait(duration=112 if restart else 6)),
    ]


def test_block_rule_carries_unmatched_conditions():
    events = []

    def start(line: Line, state: int) -> tuple[bool, int]:
        events.append(("start", state))
        return state > 0, state + 1

    def end(line: Line, state: int) -> tuple[bool, int]:
        events.append(("end", state))
        return False, state + 10

    def map_(lines: list[Line], state: int) -> tuple[Block, int]:
        events.append(("map", state))
        assert len(lines) == 2
        return [Wait(duration=state)], state

    rule = BlockRule(init_state=lambda: 0, start=start, end=end, map=map_)
    result = traverse([Wait(duration=n) for n in [4, 6, 8]], rule)
    assert events == [("start", 0), ("start", 1), ("end", 2), ("map", 12)]
    assert result == [Line.instr(Wait(duration=4)), Line.instr(Wait(duration=12))]


def test_rule_state_is_fresh_for_each_traversal():
    block = [Wait(duration=wait.MAX_WAIT + 1), Wait(duration=wait.MAX_WAIT + 1)]
    first = traverse(block, wait.long_wait)
    second = traverse(block, wait.long_wait)
    assert first == second
    assert [line.label for line in second if line.label] == ["wait0", "wait1"]


def test_line_rule_factory_creates_independent_mutable_state():
    states = []

    def init_state() -> list[int]:
        state = []
        states.append(state)
        return state

    def match(line: Line, state: list[int]) -> tuple[bool, list[int]]:
        state.append(1)
        return True, state

    def map_(line: Line, state: list[int]) -> tuple[Block, list[int]]:
        return [Wait(duration=len(state))], state

    rule = LineRule(init_state=init_state, match=match, map=map_)
    block = [Wait(duration=4), Wait(duration=6)]
    assert traverse(block, rule) == traverse(block, rule)
    assert states == [[1, 1], [1, 1]]
    assert states[0] is not states[1]
