import pytest

from qibolab.instruments.dummy import DummyDevice, DummyLocalOscillator


@pytest.fixture
def lo():
    return DummyLocalOscillator("lo", "0")


def test_oscillator_init(lo):
    assert lo.device is None
    assert lo.settings.power is None
    assert lo.settings.frequency is None
    assert lo.settings.ref_osc_source is None


def test_oscillator_connect(lo):
    assert lo.device is None
    lo.connect()
    assert lo.is_connected
    assert isinstance(lo.device, DummyDevice)
    lo.disconnect()
    assert not lo.is_connected


def test_oscillator_setup(lo):
    assert lo.frequency is None
    assert lo.power is None
    lo.setup(frequency=5e9, power=0)
    assert lo.frequency == 5e9
    assert lo.power == 0
