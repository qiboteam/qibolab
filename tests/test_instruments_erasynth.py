import numpy as np
import pytest

from qibolab import Platform
from qibolab.paths import user_folder

from .conftest import load_from_platform


@pytest.mark.qpu
def test_instruments_erasynth_init(instrument):
    assert instrument.is_connected == True
    assert instrument.device
    assert instrument.data_folder == user_folder / "instruments" / "data" / instrument.tmp_folder.name.split("/")[-1]


@pytest.mark.qpu
@pytest.mark.parametrize("instrument_name", ["ERA"])
def test_instruments_erasynth_setup(platform_name, instrument_name):
    platform = Platform(platform_name)
    settings = platform.settings
    instrument, instrument_settings = load_from_platform(platform, instrument_name)
    instrument.connect()
    instrument.setup(**settings["settings"], **instrument_settings)
    for parameter in instrument_settings:
        assert getattr(instrument, parameter) == instrument_settings[parameter]
    instrument.disconnect()


def instrument_set_and_test_parameter_values(instrument, parameter, values):
    for value in values:
        instrument._set_device_parameter(parameter, value)
        assert instrument.device.get(parameter) == value


@pytest.mark.qpu
def test_instruments_erasynth_set_device_paramter(instrument):
    instrument_set_and_test_parameter_values(
        instrument, f"power", np.arange(-60, 0, 10)
    )  # Max power is 25dBm but to be safe testing only until 0dBm
    instrument_set_and_test_parameter_values(instrument, f"frequency", np.arange(250e3, 15e9, 1e9))


@pytest.mark.qpu
def test_instruments_erasynth_start_stop_disconnect(instrument):
    instrument.start()
    instrument.stop()
    instrument.disconnect()
    assert instrument.is_connected == False
