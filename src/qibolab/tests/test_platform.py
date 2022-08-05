import pytest
from qibolab import Platform


def test_platform_multiqubit():
    from qibolab.platforms.multiqubit import MultiqubitPlatform
    platform = Platform("multiqubit")
    assert isinstance(platform, MultiqubitPlatform)
    platform = Platform("tiiq")
    assert isinstance(platform, MultiqubitPlatform)
    platform = Platform("qili")
    assert isinstance(platform, MultiqubitPlatform)


@pytest.mark.skip("Loading the platform requires access to one of the instrument's driver dll")
def test_platform_icarusq():
    from qibolab.platforms.multiqubit import MultiqubitPlatform
    platform = Platform("icarusq")
    assert isinstance(platform, MultiqubitPlatform)


def test_platform():
    with pytest.raises(RuntimeError):
        platform = Platform("nonexistent")
