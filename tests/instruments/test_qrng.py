import numpy as np
import pytest
from scipy.stats import chisquare
from serial.serialutil import SerialException

from qibolab.instruments.qrng import QRNG

P_VALUE_CUTOFF = 0.05
"""p-value cutoff for the chi-square tests."""


@pytest.fixture
def qrng():
    qrng = QRNG(address="/dev/ttyACM0")
    try:
        qrng.connect()
    except SerialException:
        pass
    return qrng


def test_random_chisquare(qrng):
    data = qrng.random(1000)

    nbins = int(np.sqrt(len(data)))
    bins = np.linspace(0, 1, nbins + 1)
    observed_frequencies, _ = np.histogram(data, bins=bins)

    expected_frequency = len(data) / nbins
    expected_frequencies = np.full(nbins, expected_frequency)
    _, p_value = chisquare(observed_frequencies, expected_frequencies)
    assert p_value > 0.05
