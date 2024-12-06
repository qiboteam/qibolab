from pydantic import TypeAdapter

from qibolab._core.instruments.qblox.ast_ import Immediate, Value


def test_immediate_load():
    load = TypeAdapter(Immediate).validate_python

    for input_ in [42, "42", "0x2a", "0o52"]:
        im = load(input_)
        assert im == 42


def test_value_load():
    load = TypeAdapter(Value).validate_python
    v = load(42)
    assert v == 42
