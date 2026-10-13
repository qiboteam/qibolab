from pydantic import TypeAdapter

from qibolab._core.instruments.qblox.q1asm.ast_ import (
    Immediate,
    Line,
    Program,
    Value,
    Wait,
)


def test_immediate_load():
    load = TypeAdapter(Immediate).validate_python

    for input_ in [42, "42", "0x2a", "0o52"]:
        im = load(input_)
        assert im == 42


def test_value_load():
    load = TypeAdapter(Value).validate_python
    v = load(42)
    assert v == 42


def test_program_from_line_sequence():
    lines = (Line.instr(Wait(duration=4)), Line.instr(Wait(duration=6)))
    program = Program.from_elements(lines)
    assert program.lines == list(lines)
