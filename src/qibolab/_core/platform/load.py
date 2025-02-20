import importlib.util
import os
from pathlib import Path
from typing import Optional, Union

from qibo.config import raise_error

from ..parameters import Hardware
from .platform import Platform

__all__ = ["create_platform", "locate_platform"]

PLATFORM = "platform.py"
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
    for path in paths:
        platform = path / name
        if platform.exists():
            return platform

    raise_error(
        ValueError,
        f"Platform {name} not found. Check ${PLATFORMS} environment variable.",
    )


def _load(platform: Path) -> Union[Platform, Hardware]:
    """Load the platform module."""
    module_name = "platform"
    spec = importlib.util.spec_from_file_location(module_name, platform / PLATFORM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create()


def locate_platform(name: str, paths: Optional[list[Path]] = None) -> Path:
    """Locate platform's path.

    The ``name`` corresponds to the name of the folder in which the platform is defined,
    i.e. the one containing the ``platform.py`` file.

    If ``paths`` are specified, the environment is ignored, and the folder search
    happens only in the specified paths.
    """
    if paths is None:
        paths = _platforms_paths()
    return _search(name, paths)


def create_platform(name: str) -> Platform:
    """A platform for executing quantum algorithms.

    It consists of a quantum processor QPU and a set of controlling instruments.

    Args:
        name (str): name of the platform.
    Returns:
        The plaform class.
    """
    if name == "dummy":
        from qibolab._core.dummy import create_dummy

        return create_dummy()

    path = _search(name, _platforms_paths())

    hardware = _load(path)
    if isinstance(hardware, Platform):
        return hardware

    return Platform.load(path, instruments=hardware.instruments, **hardware.elements)


def available_platforms() -> list[str]:
    """Returns the platforms found in the $QIBOLAB_PLATFORMS directory."""
    return [
        d.name
        for platforms in _platforms_paths()
        for d in platforms.iterdir()
        if d.is_dir()
        and Path(f"{os.environ.get(PLATFORMS)}/{d.name}/platform.py") in d.iterdir()
    ]
