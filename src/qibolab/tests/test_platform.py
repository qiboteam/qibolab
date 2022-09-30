# -*- coding: utf-8 -*-
import pytest

from qibolab import Platform


def test_platform_multiqubit():
    from qibolab.platforms.multiqubit import MultiqubitPlatform

    platform = Platform("tii5q")
    assert isinstance(platform, MultiqubitPlatform)
    platform = Platform("tii1q")
    assert isinstance(platform, MultiqubitPlatform)


def test_platform():
    with pytest.raises(RuntimeError):
        platform = Platform("nonexistent")


# TODO: test dummy platform
