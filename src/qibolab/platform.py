def Platform(name, runcard=None):
    """Platform for controlling quantum devices.

    Args:
        name (str): name of the platform. Options are 'tiiq', 'qili', 'rfsoc' and 'icarusq'.
        runcard (str): path to the yaml file containing the platform setup.

    Returns:
        The plaform class.
    """
    if not runcard:
        from os.path import exists

        from qibolab.paths import qibolab_folder

        runcard = qibolab_folder / "runcards" / f"{name}.yml"
    if name == "tii1q" or name == "tii5q" or name == "qili":
        from qibolab.platforms.multiqubit import MultiqubitPlatform as Device

        if not exists(runcard):
            from qibo.config import raise_error

            raise_error(RuntimeError, f"Runcard {name} does not exist.")

    if name == "dummy":
        from qibolab.platforms.dummy import DummyPlatform as Device
    elif name == "icarusq":
        from qibolab.platforms.icplatform import ICPlatform as Device
    elif name == "tii_rfsoc4x2":
        from qibolab.platforms.rfsoc import RFSoc1qPlatform as Device
    elif name == "dummy":
        from qibolab.platforms.dummy import DummyPlatform as Device
    else:
        from qibolab.platforms.multiqubit import MultiqubitPlatform as Device

    return Device(name, runcard)
