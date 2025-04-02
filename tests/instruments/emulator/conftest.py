from pathlib import Path

import pytest

import qibolab

HERE = Path(__file__).parent


@pytest.fixture(params=[p.name for p in (HERE / "platforms").iterdir() if p.is_dir()])
def platform(request, monkeypatch) -> qibolab.Platform:
    monkeypatch.setenv("QIBOLAB_PLATFORMS", HERE / "platforms")
    platform_name = request.param
    return qibolab.create_platform(platform_name)
