# -*- coding: utf-8 -*-
import numpy as np
import pytest
import yaml

from qibolab.instruments.rohde_schwarz import SGS100A
from qibolab.paths import qibolab_folder, user_folder

INSTRUMENTS_LIST = ["SGS100A"]
instruments = {}


@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_rohde_schwarz_init(name):
    test_runcard = qibolab_folder / "tests" / "test_instruments_rohde_schwarz.yml"
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
    assert instance.device == None
    assert instance.data_folder == user_folder / "instruments" / "data" / instance.tmp_folder.name.split("/")[-1]


@pytest.mark.xfail
@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qublox_connect(name):
    instruments[name].connect()


@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qublox_setup(name):
    if not instruments[name].is_connected:
        pytest.xfail("Instrument not available")
    else:
        test_runcard = qibolab_folder / "tests" / "test_instruments_rohde_schwarz.yml"
        with open(test_runcard, "r") as file:
            settings = yaml.safe_load(file)
        instruments[name].setup(**settings["settings"], **settings["instruments"][name]["settings"])

        for parameter in settings["instruments"][name]["settings"]:
            assert getattr(instruments[name], parameter) == settings["instruments"][name]["settings"][parameter]


def instrument_set_and_test_parameter_values(instrument, parameter, values):
    for value in values:
        instrument.set_device_parameter(parameter, value)
        assert instrument.device.get(parameter) == value


@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qublox_set_device_paramter(name):
    if not instruments[name].is_connected:
        pytest.xfail("Instrument not available")
    else:
        instrument_set_and_test_parameter_values(
            instruments[name], f"power", np.arange(-120, 0, 10)
        )  # Max power is 25dBm but to be safe testing only until 0dBm
        instrument_set_and_test_parameter_values(instruments[name], f"frequency", np.arange(1e6, 12750e6, 1e9))

        """   # TODO: add attitional paramter tests
        SGS100A:
            parameter            value
        --------------------------------------------------------------------------------
        IDN                   :	{'vendor': 'Rohde&Schwarz', 'model': 'SGS100A', 'seri...
        IQ_angle              :	None
        IQ_gain_imbalance     :	None
        IQ_impairments        :	None
        IQ_state              :	None
        I_offset              :	None
        LO_source             :	None
        Q_offset              :	None
        frequency             :	None (Hz)
        phase                 :	None (deg)
        power                 :	None (dBm)
        pulsemod_source       :	None
        pulsemod_state        :	None
        ref_LO_out            :	None
        ref_osc_external_freq :	None
        ref_osc_output_freq   :	None
        ref_osc_source        :	None
        status                :	None
        timeout               :	5 (s)
        """


@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qublox_start_stop_disconnect(name):
    if not instruments[name].is_connected:
        pytest.xfail("Instrument not available")
    else:
        instruments[name].start()
        instruments[name].stop()
        instruments[name].disconnect()
        assert instruments[name].is_connected == False
