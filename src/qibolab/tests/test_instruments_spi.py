# -*- coding: utf-8 -*-
import pytest
import yaml

from qibolab.instruments.spi import SPI
from qibolab.paths import qibolab_folder, user_folder

INSTRUMENTS_LIST = ["SPI"]
instruments = {}

# To test --> name = SpiRack
@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_spi_init(name):
    test_runcard = qibolab_folder / "tests" / "test_instruments_spi.yml"
    with open(test_runcard, "r") as file:
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
    assert instance.signature == f"{name}@{address}"
    assert (
        instance.data_folder
        == user_folder
        / "instruments"
        / "data"
        / instance.tmp_folder.name.split("/")[-1]
    )


@pytest.mark.xfail
@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_spi_connect(name):
    instruments[name].connect()


@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_spi_setup(name):
    if not instruments[name].is_connected:
        pytest.xfail("Instrument not available")
    else:
        test_runcard = qibolab_folder / "tests" / "test_instruments_spi.yml"
        with open(test_runcard, "r") as file:
            settings = yaml.safe_load(file)
        instruments[name].setup(
            **settings["settings"], **settings["instruments"][name]["settings"]
        )

        for parameter in settings["instruments"][name]["settings"]:
            assert (
                getattr(instruments[name], parameter)
                == settings["instruments"][name]["settings"][parameter]
            )


@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_spi_disconnect(name):
    if not instruments[name].is_connected:
        pytest.xfail("Instrument not available")
    else:
        instruments[name].disconnect()
        assert instruments[name].is_connected == False


@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_spi_close(name):
    if not instruments[name].is_connected:
        pytest.xfail("Instrument not available")
    else:
        instruments[name].close()
        assert instruments[name].is_connected == False
