# -*- coding: utf-8 -*-


def Platform(name, runcard=None):
    """Platform for controlling quantum devices.

    Args:
        name (str): name of the platform. Options are 'tiiq', 'qili' and 'icarusq'.
        runcard (str): path to the yaml file containing the platform setup.

    Returns:
        The plaform class.
    """
    if not runcard:
        from qibolab.paths import qibolab_folder

        runcard = qibolab_folder / "runcards" / f"{name}.yml"

    if name == "tii1q" or name == "tii5q" or name == "qili" or name == "zhinst":
        from qibolab.platforms.multiqubit import MultiqubitPlatform as Device
    elif name == "icarusq":
        from qibolab.platforms.icplatform import ICPlatform as Device
    elif name == "dummy":
        from qibolab.platforms.dummy import DummyPlatform as Device
    else:
        from qibo.config import raise_error

        raise_error(RuntimeError, f"Platform {name} is not supported.")

    return Device(name, runcard)
