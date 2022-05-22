import pytest
from qibolab import Platform


def test_platform():
    from qibolab.platforms.qbloxplatform import QBloxPlatform
    platform = Platform("tiiq")
    assert isinstance(platform, QBloxPlatform)

    from qibolab.platforms.multiqubit import MultiqubitPlatform
    platform = Platform("multiqubit")
    assert isinstance(platform, MultiqubitPlatform)

    from qibolab.platforms.icplatform import ICPlatform
    platform = Platform("icarusq")
    assert isinstance(platform, ICPlatform)

    with pytest.raises(RuntimeError):
        platform = Platform("nonexistent")
