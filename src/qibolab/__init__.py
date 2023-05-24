import importlib.metadata as im
import importlib.util
import os

import yaml
from qibo.config import raise_error

__version__ = im.version(__package__)


def create_platform(name):
    """Platform for controlling quantum devices.

    Args:
        name (str): name of the platform. Options are 'tiiq', 'qili' and 'icarusq'.
    Returns:
        The plaform class.
    """
    if name == "dummy":
        from qibolab.paths import qibolab_folder
        from qibolab.platform import create_dummy

        return create_dummy(qibolab_folder / "runcards/dummy.yml")

    profiles = os.environ.get("QIBOLAB_PLATFORMS_FILE")
    if profiles:
        if not os.path.exists(profiles):
            raise_error(RuntimeError, f"Profile file {profiles} does not exist.")
    else:
        raise_error(RuntimeError, "Please set the QIBOLAB_PLATFORMS_FILE environment variable.")

    with open(profiles) as stream:
        try:
            setup = yaml.safe_load(stream)
            platform = setup[name]
        except yaml.YAMLError:
            raise_error(yaml.YAMLError, f"Error loading {profiles} yaml file.")
        except KeyError:
            raise_error(KeyError, f"Platform {name} not found.")

    spec = importlib.util.spec_from_file_location("platform", platform)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.create()
