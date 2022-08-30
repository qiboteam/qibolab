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
        self.u3params = U3Params()

    def reload_settings(self):  # pragma: no cover
        log.info("Dummy platform does not support setting reloading.")

    def execute_pulse_sequence(self, sequence, nshots=None):  # pragma: no cover
        log.info("this is a dummy qubit that is there")
        time.sleep(self.settings.get("sleep_time"))
        ro_pulses = {pulse.qubit: pulse.serial for pulse in sequence.ro_pulses}

        results = {}
        for qubit, pulse in ro_pulses.items():
            i, q = np.random.random(2)
            results[qubit] = {pulse: (np.sqrt(i**2 + q**2), np.arctan2(q, i), i, q)}
        return results

    def run_calibration(self, show_plots=False):  # pragma: no cover
        log.info("It is a dummy qubit")
