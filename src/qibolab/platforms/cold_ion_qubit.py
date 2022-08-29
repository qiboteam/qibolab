# -*- coding: utf-8 -*-
import time

import numpy as np
import yaml
from qibo.config import log, raise_error

from qibolab.platforms.abstract import AbstractPlatform
from qibolab.u3params import U3Params


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
        nqubits = self.settings.get("nqubits")
        self.qcm = {i: DDSAD9959() for i in range(nqubits)}
        self.u3params = U3Params()

    def reload_settings(self):  # pragma: no cover
        log.info(
            "Cold ion platform does not support setting reloading in this version of the software."
        )

    def run_calibration(self, show_plots=False):  # pragma: no cover
        raise_error(NotImplementedError)

    def connect(self):
        log.info("Connecting to cold ion platform.")

    def start(self):
        log.info("Starting cold ion platform.")

    def stop(self):
        log.info("Stopping cold ion platform.")

    def disconnect(self):
        log.info("Disconnecting cold ion platform.")
