import importlib.metadata as im
import importlib.util
import os
from pathlib import Path

from qibo.config import raise_error

from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.platform import Platform

__version__ = im.version(__package__)

PLATFORMS = "QIBOLAB_PLATFORMS"


def get_platforms_path():
    """Get path to repository containing the platforms.

    Path is specified using the environment variable QIBOLAB_PLATFORMS.
    """
    profiles = os.environ.get(PLATFORMS)
    if profiles is None or not os.path.exists(profiles):
        raise_error(RuntimeError, f"Profile directory {profiles} does not exist.")
    return Path(profiles)


def create_platform(name, runcard=None):
    """A platform for executing quantum algorithms.

    It consists of a quantum processor QPU and a set of controlling instruments.

    Args:
        name (str): name of the platform. Options are 'tiiq', 'qili' and 'icarusq'.
    Returns:
        The plaform class.
    """
    if name == "dummy":
        from qibolab.dummy import create_dummy

        return create_dummy()
    if name == "dummy2":
        from qibolab.dummy2 import create_dummy2

        return create_dummy2()

    platform = get_platforms_path() / f"{name}.py"
    if not platform.exists():
        raise_error(ValueError, f"Platform {name} does not exist.")

    spec = importlib.util.spec_from_file_location("platform", platform)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if runcard is None:
        return module.create()
    return module.create(runcard)
