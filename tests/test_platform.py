import pytest

from qibolab import Platform


def test_platform_multiqubit(platform_name):
    from qibolab.platforms.multiqubit import MultiqubitPlatform

    platform = Platform(platform_name)
    assert isinstance(platform, MultiqubitPlatform)


def test_platform():
    with pytest.raises(RuntimeError):
        platform = Platform("nonexistent")


# TODO: test dummy platform
