import inspect
import os
import shutil
from pathlib import Path

import pytest

from qibolab._core.dummy.platform import create_dummy_hardware
from qibolab._core.platform import Hardware, Platform
from qibolab._core.platform.platform import PARAMETERS
from qibolab.platform import (
    PLATFORM,
    PLATFORMS_PATH,
    create_platform,
    initialize_parameters,
    load_hardware,
    locate_platform,
)


def test_create_platform(platform):
    assert isinstance(platform, Platform)


def test_create_platform_error():
    with pytest.raises(ValueError):
        _ = create_platform("nonexistent")


@pytest.fixture
def dummy_hardware(monkeypatch, tmp_path: Path) -> str:
    name = "dummy_hardware"
    parameters = initialize_parameters(hardware=create_dummy_hardware())
    platform = tmp_path / name
    shutil.copytree(Path(__file__).parent / name, platform)
    (platform / PARAMETERS).write_text(parameters.model_dump_json(indent=4))
    monkeypatch.setenv(PLATFORMS_PATH, str(tmp_path))
    return name


def test_create_platform_from_hardware(dummy_hardware: str):
    platform = create_platform(dummy_hardware)
    assert isinstance(platform, Platform)
    assert list(platform.qubits.keys()) == list(range(5))


def test_load_hardware(dummy_hardware: str):
    hw = load_hardware(dummy_hardware)
    assert isinstance(hw, Hardware)
    assert list(hw.qubits.keys()) == list(range(5))


def test_locate_platform(tmp_path: Path):
    some = tmp_path / "some"
    some.mkdir()

    for p in [some / "platform0", some / "platform1"]:
        p.mkdir()
        (p / PLATFORM).write_text("'Ciao'")

    assert locate_platform("platform0", [some]) == some / "platform0"

    with pytest.raises(ValueError):
        locate_platform("platform3")

    os.environ[PLATFORMS_PATH] = str(some)

    assert locate_platform("platform1") == some / "platform1"


def test_create_platform_multipath(tmp_path: Path):
    some = tmp_path / "some"
    others = tmp_path / "others"
    some.mkdir()
    others.mkdir()

    for p in [
        some / "platform0",
        some / "platform1",
        others / "platform1",
        others / "platform2",
    ]:
        p.mkdir()
        (p / PLATFORM).write_text(
            inspect.cleandoc(
                f"""
                from qibolab._core.platform import Platform

                def create():
                    return Platform("{p.parent.name}-{p.name}", {{}}, {{}}, {{}}, {{}})
                """
            )
        )

    os.environ[PLATFORMS_PATH] = f"{some}{os.pathsep}{others}"

    def path(name):
        return tmp_path / Path(create_platform(name).name.replace("-", os.sep))

    assert path("platform0").relative_to(some)
    assert path("platform1").relative_to(some)
    assert path("platform2").relative_to(others)
    with pytest.raises(ValueError):
        create_platform("platform3")
