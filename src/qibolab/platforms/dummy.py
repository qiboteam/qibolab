# -*- coding: utf-8 -*-
import time

import numpy as np
import yaml
from qibo.config import log, raise_error

from qibolab.platforms.abstract import AbstractPlatform


class DummyInstrument:
    # This object is used to make QCVV methods work until
    # we improve the platform abstractions
    # TODO: Remove this objects when abstractions are fixed

    def set_device_parameter(self, *args, **kwargs):
        pass


class DummyPlatform(AbstractPlatform):
    """Dummy platform that returns random voltage values.

    Useful for testing code without requiring access to hardware.

    Args:
        name (str): name of the platform.
    """

    def reload_settings(self):  # pragma: no cover
        log.info("Dummy platform does not support setting reloading.")

    def run_calibration(self, show_plots=False):  # pragma: no cover
        raise_error(NotImplementedError)

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

    def to_sequence(self, sequence, gate):  # pragma: no cover
        raise_error(NotImplementedError)

    def execute_pulse_sequence(self, sequence, nshots=None):  # pragma: no cover
        time.sleep(self.settings.get("sleep_time"))
        ro_pulses = {pulse.qubit: pulse.serial for pulse in sequence.ro_pulses}

        results = {}
        for qubit, pulse in ro_pulses.items():
            i, q = np.random.random(2)
            results[qubit] = {pulse: (np.sqrt(i**2 + q**2), np.arctan2(q, i), i, q)}
        return results
