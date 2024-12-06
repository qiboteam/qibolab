from pydantic import TypeAdapter

from qibolab._core.instruments.qblox.ast_ import Immediate, Value


def test_immediate_load():
    load = TypeAdapter(Immediate).validate_python
    im = load(42)
    assert im == 42


def test_value_load():
    load = TypeAdapter(Value).validate_python
    v = load(42)
    assert v == 42
