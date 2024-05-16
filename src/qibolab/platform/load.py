import importlib.util
import os
from pathlib import Path

from qibo.config import raise_error

from qibolab.serialize import PLATFORM

from .platform import Platform

PLATFORMS = "QIBOLAB_PLATFORMS"


def _platforms_paths() -> list[Path]:
    """Get path to repository containing the platforms.

    Path is specified using the environment variable QIBOLAB_PLATFORMS.
    """
    paths = os.environ.get(PLATFORMS)
    if paths is None:
        raise_error(RuntimeError, f"Platforms path ${PLATFORMS} unset.")

    return [Path(p) for p in paths.split(os.pathsep)]


def _search(name: str, paths: list[Path]) -> Path:
    """Search paths for given platform name."""
    for path in _platforms_paths():
        platform = path / name
        if platform.exists():
            return platform

    raise_error(
        ValueError,
        f"Platform {name} not found. Check ${PLATFORMS} environment variable.",
    )


def _load(platform: Path) -> Platform:
    """Load the platform module."""
    module_name = "platform"
    spec = importlib.util.spec_from_file_location(module_name, platform / PLATFORM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create()


def create_platform(name: str) -> Platform:
    """A platform for executing quantum algorithms.

    It consists of a quantum processor QPU and a set of controlling instruments.

    Args:
        name (str): name of the platform. Options are 'tiiq', 'qili' and 'icarusq'.
        path (pathlib.Path): path with platform serialization
    Returns:
        The plaform class.
    """
    if name == "dummy" or name == "dummy_couplers":
        from qibolab.dummy import create_dummy

        return create_dummy(with_couplers=name == "dummy_couplers")

    return _load(_search(name, _platforms_paths()))


def available_platforms() -> list[str]:
    """Returns the platforms found in the $QIBOLAB_PLATFORMS directory."""
    return [
        d.name
        for platforms in _platforms_paths()
        for d in platforms.iterdir()
        if d.is_dir()
        and Path(f"{os.environ.get(PLATFORMS)}/{d.name}/platform.py") in d.iterdir()
    ]
