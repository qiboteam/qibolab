import pytest

from qibolab import AcquisitionType as Acq
from qibolab import AveragingMode as Av

NSHOTS = 50
NSWEEP1 = 5
NSWEEP2 = 8


@pytest.fixture(params=[False, True])
def sweep(request):
    return None if request.param else []


def test_discrimination_singleshot(execute, sweep):
    result = execute(Acq.DISCRIMINATION, Av.SINGLESHOT, NSHOTS, sweep)
    if sweep == []:
        assert result.shape == (NSHOTS,)
    else:
        assert result.shape == (NSHOTS, NSWEEP1, NSWEEP2)


def test_discrimination_cyclic(execute, sweep):
    result = execute(Acq.DISCRIMINATION, Av.CYCLIC, NSHOTS, sweep)
    if sweep == []:
        assert result.shape == tuple()
    else:
        assert result.shape == (NSWEEP1, NSWEEP2)


def test_integration_singleshot(execute, sweep):
    result = execute(Acq.INTEGRATION, Av.SINGLESHOT, NSHOTS, sweep)
    if sweep == []:
        assert result.shape == (NSHOTS, 2)
    else:
        assert result.shape == (NSHOTS, NSWEEP1, NSWEEP2, 2)


def test_integration_cyclic(execute, sweep):
    result = execute(Acq.INTEGRATION, Av.CYCLIC, NSHOTS, sweep)
    if sweep == []:
        assert result.shape == (2,)
    else:
        assert result.shape == (NSWEEP1, NSWEEP2, 2)


def test_raw_singleshot(execute):
    result = execute(Acq.RAW, Av.SINGLESHOT, NSHOTS, [])
    assert result.shape == (NSHOTS, int(execute.acquisition_duration), 2)


def test_raw_cyclic(execute, sweep):
    result = execute(Acq.RAW, Av.CYCLIC, NSHOTS, sweep)
    if sweep == []:
        assert result.shape == (int(execute.acquisition_duration), 2)
    else:
        assert result.shape == (NSWEEP1, NSWEEP2, int(execute.acquisition_duration), 2)
