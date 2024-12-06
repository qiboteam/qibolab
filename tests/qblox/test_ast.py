from pydantic import TypeAdapter

from qibolab._core.instruments.qblox.ast_ import Value


def test_value_load():
    load = TypeAdapter(Value).validate_python
    v = load(42)
    assert v == 42
