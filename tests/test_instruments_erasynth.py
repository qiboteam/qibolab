import pytest

from qibolab.instruments.erasynth import ERA

from .conftest import get_instrument


@pytest.fixture(scope="module")
def era(connected_platform):
    return get_instrument(connected_platform, ERA)


@pytest.mark.qpu
def test_instruments_erasynth_connect(era):
    assert era.is_connected == True


@pytest.mark.qpu
def test_instruments_erasynth_setup(era):
    era.setup(frequency=5e9, power=-10)
    assert era.frequency == 5e9
    assert era.power == -10


@pytest.mark.qpu
def test_instruments_erasynth_start_stop_disconnect(era):
    era.start()
    era.stop()
