"""Testing result.py."""

import numpy as np
import pytest
from pytest import approx

from qibolab import AcquisitionType as Acq
from qibolab import AveragingMode as Av
from qibolab.result import magnitude, phase, probability, unpack


@pytest.mark.parametrize("result", ["iq", "raw"])
def test_polar(result, execute):
    """Testing I and Q polar representation."""
    if result == "iq":
        res = execute(Acq.INTEGRATION, Av.SINGLESHOT, 5)
    else:
        res = execute(Acq.RAW, Av.CYCLIC, 5)

    i, q = unpack(res)
    np.testing.assert_equal(np.sqrt(i**2 + q**2), magnitude(res))
    np.testing.assert_equal(np.unwrap(np.arctan2(i, q)), phase(res))


def test_probability(execute):
    """Testing raw_probability method."""
    res = execute(Acq.DISCRIMINATION, Av.SINGLESHOT, 1000)
    prob = probability(res)

    # unless the result is exactly 0, there is no need for the absolute value
    # and when its close to 0, the absolute tolerance is preventing the possible error
    # due to floating point operations
    assert prob == approx(1 - np.mean(res, axis=0))
    assert probability(res, 1) == approx(1 - prob)
