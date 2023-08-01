import pytest

from qibolab.instruments.qutech import SPI

from .conftest import get_instrument


@pytest.fixture(scope="module")
def spi(connected_platform):
    return get_instrument(connected_platform, SPI)


# To test --> name = SpiRack
@pytest.mark.qpu
def test_instruments_qutech_init(spi):
    assert spi.is_connected == True
    assert spi.device is None
    assert spi.data_folder == INSTRUMENTS_DATA_FOLDER / spi.tmp_folder.name.split("/")[-1]


@pytest.mark.qpu
def test_instruments_qutech_disconnect(spi):
    spi.connect()
    assert spi.is_connected == True
    spi.disconnect()
    assert spi.is_connected == False
    spi.connect()


@pytest.mark.qpu
def test_instruments_qutech_close(spi):
    spi.close()
    assert spi.is_connected == False
    spi.connect()
