import copy
import time

import numpy as np
from qibo.config import log, raise_error

from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence, ReadoutPulse
from qibolab.result import ExecutionResults


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

    def execute_pulse_sequence(self, sequence, nshots=None, wait_time=None):
        if wait_time is None:
            wait_time = self.settings.get("sleep_time")

        if nshots is None:
            nshots = self.settings["settings"]["hardware_avg"]

        time.sleep(wait_time)

        ro_pulses = {pulse.qubit: pulse.serial for pulse in sequence.ro_pulses}

        results = {}
        for qubit, serial in ro_pulses.items():
            i = np.random.rand(nshots)
            q = np.random.rand(nshots)
            shots = np.random.rand(nshots)
            results[qubit] = ExecutionResults.from_components(i, q, shots)
            results[serial] = copy.copy(results[qubit])
        return results

    def set_attenuation(self, qubit, att):  # pragma: no cover
        pass

    def set_current(self, qubit, current):  # pragma: no cover
        pass

    def set_gain(self, qubit, gain):  # pragma: no cover
        pass

    def sweep(self, sequence, *sweepers, nshots=1024, average=True, wait_time=None):
        results = {}
        self._sweep_recursion(
            copy.deepcopy(sequence), *sweepers, nshots=nshots, average=average, wait_time=wait_time, results=results
        )
        return results

    def _sweep_recursion(self, sequence, *sweepers, nshots=1024, average=True, wait_time=None, results=None):
        original = copy.deepcopy(sequence)
        map_old_new_pulse = {pulse: pulse.serial for pulse in sequence.ro_pulses}
        sweeper = sweepers[0]
        for value in sweeper.values:
            for pulse in copy.deepcopy(sweeper.pulses):
                shifted_pulses = []
                if sweeper.parameter == "frequency":
                    setattr(pulse, sweeper.parameter, getattr(original[pulse.qubit], sweeper.parameter) + value)
                elif sweeper.parameter == "attenuation":
                    self.set_attenuation(pulse.qubit, value)
                elif sweeper.paramter in "gain":
                    self.set_gain(pulse.qubit, value)
                else:
                    setattr(pulse, sweeper.parameter, value)
                if isinstance(pulse, ReadoutPulse):
                    map_old_new_pulse[original[pulse.qubit]] = pulse.serial

                # Add pulse with parameter shifted
                sequence.add(pulse)
                shifted_pulses.append(pulse)

            if len(sweepers) > 1:
                self._sweep_recursion(
                    sequence, *sweepers[1:], nshots=nshots, average=average, wait_time=wait_time, results=results
                )
            else:
                new_sequence = copy.deepcopy(sequence)
                # remove original pulse
                for pulse in sweeper.pulses:
                    if new_sequence.pulses.count(pulse) > 1:
                        new_sequence.pulses.remove(pulse)
                result = self.execute_pulse_sequence(new_sequence, nshots)

                # remove shifted pulses from sequence
                for shifted_pulse in shifted_pulses:
                    sequence.remove(shifted_pulse)

                # colllect result and append to original pulse
                for old, new_serial in map_old_new_pulse.items():
                    if average:
                        result[new_serial].compute_average()
                    if old.serial in results:
                        results[old.serial] += result[new_serial]
                    else:
                        results[old.serial] = result[new_serial]
                        results[old.qubit] = copy.copy(results[old.serial])
