import pytest

from qibolab import Platform
from qibolab.platforms.platform import DesignPlatform

qubit = 0


@pytest.fixture
def platform(platform_name):
    _platform = Platform(platform_name)
    if not isinstance(_platform, DesignPlatform):
        pytest.skip(f"Skipping DesignPlatform test for {_platform.name}")
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
