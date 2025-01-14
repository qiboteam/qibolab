import pytest

from qibolab._core.instruments.erasynth import ERASynth

from .conftest import get_instrument


@pytest.fixture(scope="module")
def era(connected_platform):
    return get_instrument(connected_platform, ERASynth)


@pytest.mark.qpu
def test_instruments_erasynth_connect(era):
    assert era.device is not None


@pytest.mark.qpu
def test_instruments_erasynth_setup(era):
    original_frequency = era.frequency
    original_power = era.power
    era.setup(frequency=5e9, power=-10)
    assert era.frequency == 5e9
    assert era.power == -10
    era.frequency = original_frequency
    era.power = original_power
