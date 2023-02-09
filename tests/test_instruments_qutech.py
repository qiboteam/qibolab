import pytest

from qibolab import Platform
from qibolab.paths import user_folder

from .conftest import load_from_platform


# To test --> name = SpiRack
@pytest.mark.qpu
def test_instruments_qutech_init(instrument):
    assert instrument.is_connected == True
    assert instrument.device == None
    assert instrument.data_folder == user_folder / "instruments" / "data" / instrument.tmp_folder.name.split("/")[-1]


@pytest.mark.qpu
@pytest.mark.parametrize("name", ["SPI"])
def test_instruments_qutech_setup(platform_name, name):
    platform = Platform(platform_name)
    settings = platform.settings
    instrument, instrument_settings = load_from_platform(platform, name)
    instrument.setup(**settings["settings"], **instrument_settings)


@pytest.mark.qpu
def test_instruments_qutech_disconnect(instrument):
    instrument.disconnect()
    assert instrument.is_connected == False


@pytest.mark.qpu
def test_instruments_qutech_close(instrument):
    instrument.close()
    assert instrument.is_connected == False
