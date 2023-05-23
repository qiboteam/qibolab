import yaml
import importlib.util
from qibo.config import raise_error


def build_platform(name, profiles=None):
    """Platform for controlling quantum devices.
    Args:
        name (str): name of the platform. Options are 'tiiq', 'qili' and 'icarusq'.
        profiles (str): path to the yaml file containing the platforms setup.
    Returns:
        The plaform class.
    """
    if not profiles:
        from os.path import exists
        from qibolab.paths import qibolab_folder

        profiles = qibolab_folder / "profiles.yml"
        if not exists(profiles):
            raise_error(RuntimeError, f"Profile file {profiles} does not exist.")
    
    with open(profiles, 'r') as stream:
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
