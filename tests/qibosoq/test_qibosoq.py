from pathlib import Path

import pytest

from qibolab.platform import create_platform


@pytest.fixture
def platform(monkeypatch):
    monkeypatch.setenv("QIBOLAB_PLATFORMS", Path(__file__).parent)
    return create_platform("1q")


def test_qibosoq(platform):
    pass
