import shutil
from pathlib import Path

import pytest

from qibolab._core.dummy.platform import create_dummy_hardware
from qibolab._core.platform.platform import PARAMETERS
from qibolab.platform import PLATFORMS_PATH, initialize_parameters


@pytest.fixture
def dummy_hardware(monkeypatch, tmp_path: Path) -> str:
    name = "dummy_hardware"
    parameters = initialize_parameters(hardware=create_dummy_hardware())
    platform = tmp_path / name
    shutil.copytree(Path(__file__).parent / name, platform)
    (platform / PARAMETERS).write_text(parameters.model_dump_json(indent=4))
    monkeypatch.setenv(PLATFORMS_PATH, str(tmp_path))
    return name
