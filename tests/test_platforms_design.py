import pytest

from qibolab import create_platform
from qibolab.platforms.multiqubit import MultiqubitPlatform

qubit = 0


@pytest.fixture
def platform(platform_name):
    _platform = create_platform(platform_name)
    if isinstance(_platform, MultiqubitPlatform):
        pytest.skip(f"Skipping Platform test for {_platform.name}")
    return _platform


def test_platform_lo_drive_frequency(platform):
    platform.set_lo_drive_frequency(qubit, 1e9)
    assert platform.get_lo_drive_frequency(qubit) == 1e9


def test_platform_lo_readout_frequency(platform):
    platform.set_lo_readout_frequency(qubit, 1e9)
    assert platform.get_lo_readout_frequency(qubit) == 1e9


def test_platform_attenuation(platform):
    platform.set_attenuation(qubit, 10)
    assert platform.get_attenuation(qubit) == 10


def test_platform_gain(platform):
    with pytest.raises(NotImplementedError):
        platform.set_gain(qubit, 0)
    with pytest.raises(NotImplementedError):
        platform.get_gain(qubit)


def test_platform_bias(platform):
    platform.set_bias(qubit, 0)
    assert platform.get_bias(qubit) == 0
