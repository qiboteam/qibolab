import numpy as np
import pytest
from scipy.stats import chisquare
from serial.serialutil import SerialException

from qibolab.instruments.qrng import QRNG

P_VALUE_CUTOFF = 0.05
"""p-value cutoff for the chi-square tests."""


@pytest.fixture
def qrng(mocker):
    qrng = QRNG(address="/dev/ttyACM0")
    try:
        qrng.connect()
    except SerialException:

        def extract(n):
            return np.random.randint(0, 2**qrng.extracted_bits, size=(n,))

        mocker.patch.object(qrng, "extract", side_effect=extract)
    return qrng


def normalized_chisquare(x, y):
    """Normalize frequency sums to avoid errors."""
    return chisquare(x, np.sum(x) * y / np.sum(y))


def test_random_chisquare(qrng):
    data = qrng.random(1000)

    nbins = int(np.sqrt(len(data)))
    bins = np.linspace(0, 1, nbins + 1)
    observed_frequencies, _ = np.histogram(data, bins=bins)

    expected_frequency = len(data) / nbins
    expected_frequencies = np.full(nbins, expected_frequency)
    _, p_value = normalized_chisquare(observed_frequencies, expected_frequencies)
    assert p_value > P_VALUE_CUTOFF
