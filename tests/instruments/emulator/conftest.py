from pathlib import Path

import pytest

import qibolab
from qibolab._core.instruments.emulator.engine import DynamiqsEngine, QutipEngine

HERE = Path(__file__).parent


@pytest.fixture(params=[QutipEngine(), DynamiqsEngine()])
def engine(request):
    return request.param


@pytest.fixture(params=[p.name for p in (HERE / "platforms").iterdir() if p.is_dir()])
def platform(request, engine, monkeypatch) -> qibolab.Platform:
    monkeypatch.setenv("QIBOLAB_PLATFORMS", HERE / "platforms")
    platform_name = request.param
    platform = qibolab.create_platform(platform_name)
    # engine is stored inside a dictionary (instruments) under different keys
    for v in platform.instruments.values():
        v.engine = engine
    return platform
