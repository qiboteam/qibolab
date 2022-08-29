# -*- coding: utf-8 -*-
import time

import numpy as np
import yaml
from qibo.config import log, raise_error

from qibolab.platforms.abstract import AbstractPlatform
from qibolab.u3params import U3Params


class DDSAD9959:
    # This object is used to make QCVV methods work until
    # we improve the platform abstractions
    # TODO: Remove this objects when abstractions are fixed

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

        # create dummy instruments
        nqubits = self.settings.get("nqubits")
        # TODO: Remove these when platform abstraction is fixed
        self.qcm = {i: DDSAD9959() for i in range(nqubits)}
        self.qrm = {i: DDSAD9959() for i in range(nqubits)}

        from qibolab.u3params import U3Params

        self.u3params = U3Params()

    def reload_settings(self):  # pragma: no cover
        log.info("Dummy platform does not support setting reloading.")

    def connect(self):
        log.info("Connecting to dummy platform.")

    def setup(self):
        log.info("Setting up dummy platform.")

    def start(self):
        log.info("Starting dummy platform.")

    def stop(self):
        log.info("Stopping dummy platform.")

    def disconnect(self):
        log.info("Disconnecting dummy platform.")
