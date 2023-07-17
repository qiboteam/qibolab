import numpy as np
import pytest

from qibolab.instruments.erasynth import ERA

from .conftest import get_instrument


@pytest.fixture(scope="module")
def era(platform):
    return get_instrument(platform, ERA)


@pytest.mark.qpu
def test_instruments_erasynth_init(era):
    era.connect()
    assert era.is_connected == True
    assert era.device
    era.disconnect()


@pytest.mark.qpu
def test_instruments_erasynth_setup(era):
    era.connect()
    era.setup(frequency=5e9, power=-10)
    assert era.frequency == 5e9
    assert era.power == -10
    instrument.disconnect()


def set_and_test_parameter_values(instrument, parameter, values):
    for value in values:
        instrument._set_device_parameter(parameter, value)
        assert instrument.device.get(parameter) == value


@pytest.mark.qpu
def test_instruments_erasynth_set_device_paramter(era):
    era.connect()
    set_and_test_parameter_values(
        era, f"power", np.arange(-60, 0, 10)
    )  # Max power is 25dBm but to be safe testing only until 0dBm
    set_and_test_parameter_values(era, f"frequency", np.arange(250e3, 15e9, 1e9))
    era.disconnect()


@pytest.mark.qpu
def test_instruments_erasynth_start_stop_disconnect(era):
    era.connect()
    era.start()
    era.stop()
    era.disconnect()
    assert era.is_connected == False
