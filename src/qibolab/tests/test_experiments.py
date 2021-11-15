import pytest
import qibolab


def test_experiment_getter_setter():
    assert qibolab.get_experiment() == "icarusq"
    with pytest.raises(OSError): # error raised because a file is not found
        qibolab.set_experiment("awg")
    with pytest.raises(KeyError):
        qibolab.set_experiment("test")
    qibolab.set_experiment("icarusq")
