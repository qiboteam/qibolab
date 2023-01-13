import time

import numpy as np
import yaml
from qibo.config import log, raise_error

from qibolab.platforms.abstract import AbstractPlatform


class DummyPlatform(AbstractPlatform):
    """Dummy platform that returns random voltage values.

    Useful for testing code without requiring access to hardware.

    Args:
        name (str): name of the platform.
    """

    def __init__(self, name, runcard):
        super().__init__(name, runcard)

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
            results[pulse] = (np.sqrt(i**2 + q**2), np.arctan2(q, i), i, q)
            results[qubit] = (np.sqrt(i**2 + q**2), np.arctan2(q, i), i, q)
            # results[qubit] = {pulse: (np.sqrt(i**2 + q**2), np.arctan2(q, i), i, q)}
        return results

    def get_resonator_frequency(self, qubit):
        return self.characterization["single_qubit"][qubit]["resonator_freq"]

    def set_resonator_frequency(self, qubit, freq):
        self.characterization["single_qubit"][qubit]["resonator_freq"] = freq

    def set_lo_frequency(self, qubit, freq):
        pass

    def set_attenuation(self, qubit, att):
        pass
