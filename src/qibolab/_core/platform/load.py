import importlib.util
import os
from pathlib import Path

from .components import Hardware
from .platform import Platform

__all__ = ["create_platform"]

PLATFORM = "platform.py"
"""Conventional name of the file containing the platform definition.

This is only used for the builtin platform-retrieval mechanism.

.. tip::

    A Qibolab platform can be defined completely dynamically. Or it can be loaded
    in any other way, from custom-named files.
    However, in this case, no support for lookup and loading of the platforms is
    provided, and the user is responsible to instantiate a full :class:`Platform`
    object, with all the parameters value set as intended (cf.
    :class:`qibolab.Parameters`).

"""

PLATFORMS_PATH = "QIBOLAB_PLATFORMS"
"""Environment variable where to store the platforms path.

This is intended to be a ``:``-separated list of paths, which are searched in order for
folders containing platforms, identified by the presence of a source file named as
specified by :const:`PLATFORM`.
The paths appearing before in the list take priority over the following ones.
"""


def _platforms_paths() -> list[Path]:
    """Get path to repository containing the platforms.

    Path is specified using the environment variable QIBOLAB_PLATFORMS.
    """
    paths = os.environ.get(PLATFORMS_PATH)
    if paths is None:
        raise RuntimeError(f"Platforms path ${PLATFORMS_PATH} unset.")

    return [Path(p) for p in paths.split(os.pathsep)]


def _search(name: str, paths: list[Path]) -> Path:
    """Search paths for given platform name."""
    for path in paths:
        platform = path / name
        if platform.exists():
            return platform

    raise ValueError(
        f"Platform {name} not found. Check ${PLATFORMS_PATH} environment variable.",
    )


def evaluate_path(name: str | os.PathLike[str]) -> Path:
    """Search path based on string, or use it literally."""
    # just appending the CWD to the platforms path does the job
    # - relative paths are interpreted relative to the current directory, which is the
    #   intended behavioe
    # - absolute path are taken as absolute anyhow
    return _search(str(name), [Path.cwd()] + _platforms_paths())


def locate_platform(name: str, paths: list[Path] | None = None) -> Path:
    """Locate platform's path.

    The ``name`` corresponds to the name of the folder in which the platform is defined,
    i.e. the one containing the ``platform.py`` file.

    If ``paths`` are specified, the environment is ignored, and the folder search
    happens only in the specified paths.
    """
    if paths is None:
        paths = _platforms_paths()
    return _search(name, paths)


def load_platform(platform: Path) -> Platform | Hardware:
    """Load the platform module."""
    spec = importlib.util.spec_from_file_location("platform", platform / PLATFORM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create()


def load_hardware(name: str | os.PathLike[str]) -> Hardware:
    """Load the hardware representation from platform.

    It loads the :class:`Hardware` given either a :class:`str` representing its
    name, or a path to the Python module containing it.
    """
    path = evaluate_path(name)
    hardware = load_platform(path)
    assert isinstance(hardware, Hardware)
    return hardware


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

    hardware = load_platform(path)
    if isinstance(hardware, Platform):
        return hardware

    return Platform.load(path, **vars(hardware))


def available_platforms() -> list[str]:
    """Returns the platforms found in the $QIBOLAB_PLATFORMS directory."""
    return [
        d.name
        for platforms in _platforms_paths()
        for d in platforms.iterdir()
        if d.is_dir()
        and Path(f"{os.environ.get(PLATFORMS_PATH)}/{d.name}/{PLATFORM}") in d.iterdir()
    ]
