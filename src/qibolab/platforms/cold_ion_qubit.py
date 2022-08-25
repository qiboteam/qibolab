import time
import numpy as np
import yaml
from qibo.config import log, raise_error
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.u3params import U3Params

# TODO: Implement the platform for the initialisation of the qubit using the cold ion platform


class DDSAD9959:
    def set_device_parameter(self, *args, **kwargs):
        pass

class cold_ion(AbstractPlatform):
    """Platform for controlling quantum devices with cold ion platform.

    Example:
        .. code-block:: python

            from qibolab import Platform

            platform = Platform("cold_ion")

    """

    def __init__(self, name, runcard):
        self.name = name
        self.runcard = runcard
        self.is_connected = False
        # Load platform settings
        with open(runcard, "r") as file:
            self.settings = yaml.safe_load(file)


    def reload_settings(self): 
        log.info("Cold ion platform does not support setting reloading in this version of the software.")

    def run_calibration(self, show_plots=False):  
        raise_error(NotImplementedError)

    def connect(self):
        log.info("Connecting to cold ion platform.") 
        raise_error(NotImplementedError)

    def start(self):
        log.info("Starting cold ion platform.")  
        raise_error(NotImplementedError)

    def stop(self):
        log.info("Stopping cold ion platform.")  
        raise_error(NotImplementedError)

    def disconnect(self):
        log.info("Disconnecting cold ion platform.") 
        raise_error(NotImplementedError)


