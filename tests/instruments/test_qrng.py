import numpy as np
import pytest
from serial.serialutil import SerialException

from qibolab.instruments.qrng import QRNG

RAW_BITS = 12
"""Number of bits in each QRNG sample."""


@pytest.fixture
def qrng(mocker):
    qrng = QRNG(address="/dev/ttyACM0")
    try:
        qrng.connect()
    except SerialException:

        def read(n):
            return list(np.random.randint(0, 2**RAW_BITS, size=(n,)))

        mocker.patch.object(qrng, "read", side_effect=read)
    return qrng


def test_qrng_random(qrng):
    data = qrng.random(1000)
    assert isinstance(data, np.ndarray)
    assert data.shape == (1000,)
