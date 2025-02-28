import numpy as np
import pytest
from serial.serialutil import SerialException

from qibolab.instruments.qrng import QRNG, ShaExtractor, ToeplitzExtractor

RAW_BITS = 12
"""Number of bits in each QRNG sample."""


@pytest.fixture
def extractor():
    return ShaExtractor()


class MockPort:
    """Mock the serial port when QRNG device is not available for testing."""

    def read(self, n: int) -> str:
        data = []
        for i in range(n):
            if i % 4 == 3:
                data.append(" ")
            else:
                data.append(str(np.random.randint(0, 10)))
        return "".join(data).encode("utf-8")


@pytest.fixture
def qrng(extractor):
    qrng = QRNG(address="/dev/ttyACM0", extractor=extractor)
    try:
        qrng.connect()
    except SerialException:
        qrng.port = MockPort()
    return qrng


@pytest.mark.parametrize("extractor", [ShaExtractor(), ToeplitzExtractor()])
def test_qrng_random(qrng):
    data = qrng.random(1000)
    assert isinstance(data, np.ndarray)
    assert data.shape == (1000,)
