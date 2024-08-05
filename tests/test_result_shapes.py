import pytest

from qibolab import AcquisitionType as Acq
from qibolab import AveragingMode as Av

NSHOTS = 50
NSWEEP1 = 5
NSWEEP2 = 8


@pytest.mark.parametrize("sweep", [False, True])
def test_discrimination_singleshot(execute, sweep):
    result = execute(Acq.DISCRIMINATION, Av.SINGLESHOT, sweep, NSHOTS)
    if sweep:
        assert result.shape == (NSHOTS, NSWEEP1, NSWEEP2)
    else:
        assert result.shape == (NSHOTS,)


@pytest.mark.parametrize("sweep", [False, True])
def test_discrimination_cyclic(execute, sweep):
    result = execute(Acq.DISCRIMINATION, Av.CYCLIC, sweep, NSHOTS)
    if sweep:
        assert result.shape == (NSWEEP1, NSWEEP2)
    else:
        assert result.shape == tuple()


@pytest.mark.parametrize("sweep", [False, True])
def test_integration_singleshot(execute, sweep):
    result = execute(Acq.INTEGRATION, Av.SINGLESHOT, sweep, NSHOTS)
    if sweep:
        assert result.shape == (NSHOTS, NSWEEP1, NSWEEP2, 2)
    else:
        assert result.shape == (NSHOTS, 2)


@pytest.mark.parametrize("sweep", [False, True])
def test_integration_cyclic(execute, sweep):
    result = execute(Acq.INTEGRATION, Av.CYCLIC, sweep, NSHOTS)
    if sweep:
        assert result.shape == (NSWEEP1, NSWEEP2, 2)
    else:
        assert result.shape == (2,)
