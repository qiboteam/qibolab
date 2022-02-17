import pathlib
from qibo.config import raise_error


def Platform(name, runcard=None):
    """Platform for controlling quantum devices.

    Args:
        name (str): name of the platform. Options are 'tiiq', 'qili' and 'icarusq'.
        runcard (str): path to the yaml file containing the platform setup.

    Returns:
        The plaform class.
    """
    if not runcard:
        runcard = pathlib.Path(__file__).parent / "runcards" / f"{name}.yml"
    if name == 'tiiq' or name == 'qili':
        from qibolab.platforms.qbloxplatform import QBloxPlatform as Device
    elif name == 'icarusq':
        from qibolab.platforms.icplatform import ICPlatform as Device
    else:
        raise_error(RuntimeError, f"Platform {name} is not supported.")
    return Device(name, runcard)

