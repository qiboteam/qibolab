import copy
import time

import numpy as np
from qibo.config import log, raise_error

from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence, ReadoutPulse
from qibolab.result import ExecutionResults
from qibolab.sweeper import Parameter


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

    def execute_pulse_sequence(self, sequence, nshots=None, relaxation_time=None):
        if relaxation_time is None:
            relaxation_time = self.settings.get("sleep_time")

        if nshots is None:
            nshots = self.settings["settings"]["hardware_avg"]

        time.sleep(relaxation_time)

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

    def get_attenuation(self, qubit):  # pragma: no cover
        pass

    def get_current(self, qubit):  # pragma: no cover
        pass

    def get_gain(self, qubit):  # pragma: no cover
        pass

    def sweep(self, sequence, *sweepers, nshots=1024, average=True, relaxation_time=None):
        results = {}
        sweeper_pulses = {}

        # create copy of the sequence
        copy_sequence = copy.deepcopy(sequence)

        # create dictionary containing pulses for each sweeper that point to the same original sequence
        # which is copy_sequence
        for sweeper in sweepers:
            sweeper_pulses[sweeper.parameter] = {
                pulse.serial: pulse for pulse in copy_sequence if pulse in sweeper.pulses
            }

        # perform sweeping recursively
        self._sweep_recursion(
            copy_sequence,
            copy.deepcopy(sequence),
            *sweepers,
            nshots=nshots,
            average=average,
            relaxation_time=relaxation_time,
            results=results,
            sweeper_pulses=sweeper_pulses,
        )
        return results

    def _sweep_recursion(
        self,
        sequence,
        original_sequence,
        *sweepers,
        nshots=1024,
        average=True,
        relaxation_time=None,
        results=None,
        sweeper_pulses=None,
    ):
        map_original_shifted = {pulse: pulse.serial for pulse in original_sequence.ro_pulses}
        sweeper = sweepers[0]

        # store values before starting to sweep
        original_value = self._save_original_value(sweeper, sweeper_pulses)

        # perform sweep recursively
        for value in sweeper.values:
            self._update_pulse_sequence_parameters(
                sweeper, sweeper_pulses, original_sequence, map_original_shifted, value
            )
            if len(sweepers) > 1:
                self._sweep_recursion(
                    sequence,
                    original_sequence,
                    *sweepers[1:],
                    nshots=nshots,
                    average=average,
                    relaxation_time=relaxation_time,
                    results=results,
                    sweeper_pulses=sweeper_pulses,
                )
            else:
                new_sequence = copy.deepcopy(sequence)
                result = self.execute_pulse_sequence(new_sequence, nshots)

                # colllect result and append to original pulse
                for original_pulse, new_serial in map_original_shifted.items():
                    acquisition = result[new_serial].compute_average() if average else result[new_serial]

                    if results:
                        results[original_pulse.serial] += acquisition
                    else:
                        results[original_pulse.serial] = acquisition
                        results[original_pulse.qubit] = copy.copy(results[original_pulse.serial])

        # restore initial value of the pul
        self._restore_initial_value(sweeper, sweeper_pulses, original_value)

    def _save_original_value(self, sweeper, sweeper_pulses):
        """Helper method for _sweep_recursion"""
        original_value = {}
        pulses = sweeper_pulses[sweeper.parameter]
        # save original value of the parameter swept
        for pulse in pulses:
            if sweeper.parameter not in [Parameter.attenuation, Parameter.gain, Parameter.current]:
                original_value[pulse] = getattr(pulses[pulse], sweeper.parameter.name)

        return original_value

    def _restore_initial_value(self, sweeper, sweeper_pulses, original_value):
        """Helper method for _sweep_recursion"""
        pulses = sweeper_pulses[sweeper.parameter]
        for pulse in pulses:
            if sweeper.parameter not in [Parameter.attenuation, Parameter.gain, Parameter.current]:
                setattr(pulses[pulse], sweeper.parameter.name, original_value[pulse])

    def _update_pulse_sequence_parameters(
        self, sweeper, sweeper_pulses, original_sequence, map_original_shifted, value
    ):
        """Helper method for _sweep_recursion"""
        pulses = sweeper_pulses[sweeper.parameter]
        for pulse in pulses:
            if sweeper.parameter is Parameter.frequency:
                if isinstance(pulses[pulse], ReadoutPulse):
                    value += self.qubits[pulses[pulse].qubit].readout_frequency
                else:
                    value += self.qubits[pulses[pulse].qubit].drive_frequency
                setattr(pulses[pulse], sweeper.parameter.name, value)
            elif sweeper.parameter is Parameter.amplitude:
                current_amplitude = getattr(pulses[pulse], sweeper.parameter.name)
                setattr(pulses[pulse], sweeper.parameter.name, int(current_amplitude * value))
            elif sweeper.parameter is Parameter.attenuation:
                self.set_attenuation(pulses[pulse].qubit, value)
            elif sweeper.parameter is Parameter.gain:
                self.set_gain(pulses[pulse].qubit, value)
            elif sweeper.parameter is Parameter.current:
                self.set_current(pulses[pulse].qubit, value)
            else:
                setattr(pulses[pulse], sweeper.parameter.name, value)
            if isinstance(pulses[pulse], ReadoutPulse):
                map_original_shifted[original_sequence[pulses[pulse].qubit]] = pulses[pulse].serial
