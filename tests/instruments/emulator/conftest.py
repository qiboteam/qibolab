from pathlib import Path

import pytest

import qibolab

HERE = Path(__file__).parent


@pytest.fixture
def platform(monkeypatch) -> qibolab.Platform:
    monkeypatch.setenv("QIBOLAB_PLATFORMS", HERE.parent)
    return qibolab.create_platform(HERE.name)
