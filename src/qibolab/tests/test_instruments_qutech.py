import pytest

from qibolab import Platform
from qibolab.instruments.qutech import SPI
from qibolab.paths import user_folder

INSTRUMENTS_LIST = ["SPI"]


class InstrumentsDict(dict):
    def __getitem__(self, name):
        if name not in self:
            pytest.skip(f"Skip {name} test as it is not included in the tested platforms.")
        else:
            return super().__getitem__(name)


instruments = InstrumentsDict()

# To test --> name = SpiRack
@pytest.mark.qpu
@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qutech_init(platform_name, name):
    settings = Platform(platform_name).settings

    if name not in settings["instruments"]:
        pytest.skip(f"Skip {name} test as it is not included in the tested platforms.")

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
def test_instruments_qutech_setup(platform_name, name):
    settings = Platform(platform_name).settings
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
