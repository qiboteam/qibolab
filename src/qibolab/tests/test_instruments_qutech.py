import pytest
import yaml

from qibolab.instruments.qutech import SPI
from qibolab.paths import qibolab_folder, user_folder

INSTRUMENTS_LIST = ["SPI"]
instruments = {}

# To test --> name = SpiRack
@pytest.mark.qpu
@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qutech_init(name):
    test_runcard = qibolab_folder / "tests" / "test_instruments_qutech.yml"
    with open(test_runcard) as file:
        settings = yaml.safe_load(file)

    # Instantiate instrument
    lib = settings["instruments"][name]["lib"]
    i_class = settings["instruments"][name]["class"]
    address = settings["instruments"][name]["address"]
    from importlib import import_module

    InstrumentClass = getattr(import_module(f"qibolab.instruments.{lib}"), i_class)
    instance = InstrumentClass(name, address)
    instruments[name] = instance
    assert instance.name == name
    assert instance.address == address
    assert instance.is_connected == False
    assert instance.device == None
    assert instance.signature == f"{name}@{address}"
    assert instance.data_folder == user_folder / "instruments" / "data" / instance.tmp_folder.name.split("/")[-1]


@pytest.mark.qpu
@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qutech_connect(name):
    instruments[name].connect()


@pytest.mark.qpu
@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qutech_setup(name):
    test_runcard = qibolab_folder / "tests" / "test_instruments_qutech.yml"
    with open(test_runcard) as file:
        settings = yaml.safe_load(file)
    instruments[name].setup(**settings["settings"], **settings["instruments"][name]["settings"])


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
