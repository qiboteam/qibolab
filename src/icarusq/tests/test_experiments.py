import pytest
import icarusq


def test_experiment_getter_setter():
    assert icarusq.get_experiment() == "icarusq"
    with pytest.raises(OSError): # error raised because a file is not found
        icarusq.set_experiment("awg")
    with pytest.raises(KeyError):
        icarusq.set_experiment("test")
    icarusq.set_experiment("icarusq")
