import copy
import time

import numpy as np
from qibo.config import log, raise_error

from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import ReadoutPulse
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

    def execute_pulse_sequence(self, sequence, nshots=None):  # pragma: no cover
        time.sleep(self.settings.get("sleep_time"))

        if nshots is None:
            nshots = self.settings["settings"]["hardware_avg"]

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

    def sweep(self, sequence, *sweepers, nshots=1024, average=True):
        original = copy.deepcopy(sequence)
        map_old_new_pulse = {pulse: pulse.serial for pulse in sequence.ro_pulses}
        results = {}
        if len(sweepers) == 1:
            sweeper = sweepers[0]
            sweeper_pulses = copy.deepcopy(sweeper.pulses)
            for value in sweeper.values:
                for pulse in sweeper_pulses:
                    sequence.remove(pulse)
                    if sweeper.parameter == "amplitude" and max(sweeper.values) > 1:
                        self.set_attenuation(pulse.qubit, value)
                    else:
                        setattr(pulse, sweeper.parameter, getattr(original[pulse.qubit], sweeper.parameter) + value)
                    if isinstance(pulse, ReadoutPulse):
                        map_old_new_pulse[original[pulse.qubit]] = pulse.serial

                    sequence.add(pulse)

                result = self.execute_pulse_sequence(sequence, nshots)

                for old, new_serial in map_old_new_pulse.items():
                    result[new_serial].i = result[new_serial].i.mean()
                    result[new_serial].q = result[new_serial].q.mean()
                    if old.serial in results:
                        results[old.serial] += result[new_serial]
                    else:
                        results[old.serial] = result[new_serial]
                        results[old.qubit] = copy.copy(results[old.serial])

        elif len(sweepers) == 2:
            for value1 in sweepers[0].values:
                for value2 in sweepers[1].values:
                    for sweeper in sweepers:
                        for pulse in sweeper.pulses:
                            value = value1 if sweeper == sweepers[0] else value2
                            if sweeper.parameter == "amplitude" and max(sweeper.values) > 1:
                                self.set_attenuation(pulse.qubit, value)
                            else:
                                setattr(
                                    pulse, sweeper.parameter, getattr(original[pulse.qubit], sweeper.parameter) + value
                                )

                    result = self.execute_pulse_sequence(sequence, nshots)

                    for new_pulse in set(sweepers[0].pulses + sweepers[1].pulses):
                        for pulse in original:
                            if all(
                                [
                                    test_changed(pulse, new_pulse, sweeper, value)
                                    for sweeper, value in zip([sweepers[0], sweepers[1]], [value1, value2])
                                ]
                            ):
                                # if getattr(pulse, sweepers[0].parameter) + value1 == getattr(new_pulse, sweepers[0].parameter) and getattr(pulse, sweepers[1].parameter) + value2 == getattr(new_pulse, sweepers[1].parameter):
                                result[new_pulse.serial].i = result[new_pulse.serial].i.mean()
                                result[new_pulse.serial].q = result[new_pulse.serial].q.mean()
                                if pulse.serial in results:
                                    results[pulse.serial] += result[new_pulse.serial]
                                else:
                                    results[pulse.serial] = result[new_pulse.serial]
                                results[pulse.qubit] = results[pulse.serial]
        else:
            raise_error("Dummy platform supports can support up to 2 sweepers.")

        return result
