import pytest

from qibolab import Platform
from qibolab.instruments.qutech import SPI
from qibolab.paths import user_folder

from .utils import InstrumentsDict, load_from_platform

INSTRUMENTS_LIST = ["SPI"]


instruments = InstrumentsDict()
instruments_settings = {}


# To test --> name = SpiRack
@pytest.mark.qpu
@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qutech_init(platform_name, name):
    settings = Platform(platform_name).settings
    # Instantiate instrument
    instance, instr_settings = load_from_platform(settings, name)
    instruments[name] = instance
    instruments_settings[name] = instr_settings
    assert instance.name == name
    assert instance.is_connected == False
    assert instance.device == None
    assert instance.data_folder == user_folder / "instruments" / "data" / instance.tmp_folder.name.split("/")[-1]


@pytest.mark.qpu
@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qutech_connect(name):
    instruments[name].connect()


@pytest.mark.qpu
@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qutech_setup(platform_name, name):
    settings = Platform(platform_name).settings
    instruments[name].setup(**settings["settings"], **instruments_settings[name])


@pytest.mark.qpu
@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qutech_disconnect(name):
    instruments[name].disconnect()
    assert instruments[name].is_connected == False


@pytest.mark.qpu
@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qutech_close(name):
    instruments[name].close()
    assert instruments[name].is_connected == False
