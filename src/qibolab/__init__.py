import importlib.metadata as im
import importlib.util
import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from qibo.config import raise_error

__version__ = im.version(__package__)

PROFILE = "QIBOLAB_PLATFORMS_FILE"


class Profile:
    def __init__(self, path: Path):
        profile = tomllib.loads(path.read_text(encoding="utf-8"))

        paths = {}
        for name, p in profile["paths"].items():
            paths[name] = path.parent / Path(p)

        self.paths = paths


def create_platform(name, runcard=None):
    """Platform for controlling quantum devices.

    Args:
        name (str): name of the platform. Options are 'tiiq', 'qili' and 'icarusq'.
    Returns:
        The plaform class.
    """
    if name == "dummy":
        from qibolab.paths import qibolab_folder
        from qibolab.platform import create_dummy

        return create_dummy(qibolab_folder / "runcards" / "dummy.yml")

    profiles = Path(os.environ.get(PROFILE))
    if not os.path.exists(profiles):
        raise_error(RuntimeError, f"Profile file {profiles} does not exist.")

    platform = Profile(profiles).paths[name]

    spec = importlib.util.spec_from_file_location("platform", platform)
    if spec is None:
        raise ModuleNotFoundError
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if runcard is None:
        return module.create()
    return module.create(runcard)
