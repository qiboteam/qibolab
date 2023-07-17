import pytest

from qibolab.instruments.erasynth import ERA

from .conftest import get_instrument


@pytest.fixture(scope="module")
def era(platform):
    return get_instrument(platform, ERA)


@pytest.mark.qpu
def test_instruments_erasynth_connect(era):
    era.connect()
    assert era.is_connected == True
    era.disconnect()


@pytest.mark.qpu
def test_instruments_erasynth_setup(era):
    era.connect()
    era.setup(frequency=5e9, power=-10)
    assert era.frequency == 5e9
    assert era.power == -10
    era.disconnect()


@pytest.mark.qpu
def test_instruments_erasynth_start_stop_disconnect(era):
    era.connect()
    era.start()
    era.stop()
    era.disconnect()
    assert era.is_connected == False
