import pytest

from qibolab import Platform
from qibolab.platforms.abstract import AbstractPlatform


def test_platform_multiqubit(platform_name):
    platform = Platform(platform_name)
    assert isinstance(platform, AbstractPlatform)


def test_platform():
    with pytest.raises(RuntimeError):
        platform = Platform("nonexistent")


# TODO: test dummy platform
