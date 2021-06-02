import pytest
import qiboicarusq
import numpy as np


def test_experiment_getter_setter():
    assert qiboicarusq.get_experiment() == "icarusq"
    with pytest.raises(OSError): # error raised because a file is not found
        qiboicarusq.set_experiment("awg")
    with pytest.raises(KeyError):
        qiboicarusq.set_experiment("test")
    qiboicarusq.set_experiment("icarusq")
