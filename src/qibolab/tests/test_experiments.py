import pytest
import qibolab
import pyvisa


def test_experiment_getter_setter():
    assert qibolab.get_experiment() == "icarusq"
    with pytest.raises(KeyError):
        qibolab.set_experiment("test")
    qibolab.set_experiment("icarusq")


@pytest.mark.xfail(raises=pyvisa.errors.VisaIOError)
def test_icarusq_awg_setter():
    assert qibolab.get_experiment() == "icarusq"
    qibolab.set_experiment("awg")
    qibolab.set_experiment("icarusq")
