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
            results[serial] = copy.copy.copy(results[qubit])
        return results

    def set_attenuation(self, qubit, att):
        """Empty since a dummy platform is not connected to any instrument."""

    def set_bias(self, qubit, bias):
        """Empty since a dummy platform is not connected to any instrument."""

    def set_gain(self, qubit, gain):
        """Empty since a dummy platform is not connected to any instrument."""

    def get_attenuation(self, qubit):
        """Empty since a dummy platform is not connected to any instrument."""

    def get_bias(self, qubit):
        """Empty since a dummy platform is not connected to any instrument."""

    def get_gain(self, qubit):
        """Empty since a dummy platform is not connected to any instrument."""

    def sweep(self, sequence, *sweepers, nshots=1024, average=True, wait_time=None):
        results = {}
        sweeper_pulses = {}

        # create copy of the sequence
        copy_sequence = copy.deepcopy(sequence)
        map_original_shifted = {pulse: pulse.serial for pulse in copy.deepcopy(copy_sequence).ro_pulses}

        # create dictionary containing pulses for each sweeper that point to the same original sequence
        # which is copy_sequence
        for sweeper in sweepers:
            if sweeper.pulses is not None:
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
            wait_time=wait_time,
            results=results,
            sweeper_pulses=sweeper_pulses,
            map_original_shifted=map_original_shifted
        )
        return results

    def set_attenuation(self, qubit, att):  # pragma: no cover
        pass

    def set_current(self, qubit, current):  # pragma: no cover
        pass

        if sweeper.qubits is not None:
            for qubit in sweeper.qubits:
                if sweeper.parameter is Parameter.attenuation:
                    self.set_attenuation(qubit, value)
                elif sweeper.parameter is Parameter.gain:
                    self.set_gain(qubit, value)
                elif sweeper.parameter is Parameter.bias:
                    self.set_bias(qubit, value)

    def get_attenuation(self, qubit):  # pragma: no cover
        pass

    def get_current(self, qubit):  # pragma: no cover
        pass

    def get_gain(self, qubit):  # pragma: no cover
        pass

    def sweep(self, sequence, *sweepers, nshots=1024, average=True, wait_time=None):
        results = {}
        map_sweepers = {}
        copy_sequence = copy.deepcopy(sequence)
        for sweeper in sweepers:
            map_sweepers[sweeper.parameter] = {
                pulse.serial: pulse for pulse in copy_sequence if pulse in sweeper.pulses
            }

        self._sweep_recursion(
            copy_sequence,
            copy.deepcopy(sequence),
            *sweepers,
            nshots=nshots,
            average=average,
            wait_time=wait_time,
            results=results,
            map_sweepers=map_sweepers
        )
        return results

    def _sweep_recursion(
        self,
        sequence,
        original_sequence,
        *sweepers,
        nshots=1024,
        average=True,
        wait_time=None,
        results=None,
        map_sweepers=None
    ):
        map_original_shifted = {pulse: pulse.serial for pulse in original_sequence.ro_pulses}
        original_value = {}
        sweeper = sweepers[0]
        map_sweeper_to_copy = map_sweepers[sweeper.parameter]
        # save original value of the parameter swept
        for pulse in map_sweeper_to_copy:
            if sweeper.parameter not in [Parameter.attenuation, Parameter.gain, Parameter.current]:
                original_value[pulse] = getattr(map_sweeper_to_copy[pulse], sweeper.parameter.name)

        # perform sweep recursively
        for value in sweeper.values:
            for pulse in map_sweeper_to_copy:
                if sweeper.parameter is Parameter.frequency:
                    if isinstance(map_sweeper_to_copy[pulse], ReadoutPulse):
                        value += self.qubits[map_sweeper_to_copy[pulse].qubit].readout_frequency
                    else:
                        value += self.qubits[map_sweeper_to_copy[pulse].qubit].drive_frequency
                    setattr(map_sweeper_to_copy[pulse], sweeper.parameter.name, value)
                elif sweeper.parameter is Parameter.attenuation:
                    self.set_attenuation(map_sweeper_to_copy[pulse].qubit, value)
                elif sweeper.parameter is Parameter.gain:
                    self.set_gain(map_sweeper_to_copy[pulse].qubit, value)
                else:
                    setattr(map_sweeper_to_copy[pulse], sweeper.parameter.name, value)
                if isinstance(map_sweeper_to_copy[pulse], ReadoutPulse):
                    map_original_shifted[original_sequence[map_sweeper_to_copy[pulse].qubit]] = map_sweeper_to_copy[
                        pulse
                    ].serial
            if len(sweepers) > 1:
                self._sweep_recursion(
                    sequence,
                    original_sequence,
                    *sweepers[1:],
                    nshots=nshots,
                    average=average,
                    wait_time=wait_time,
                    results=results,
                    map_sweepers=map_sweepers
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

        # restore parameter value:
        for pulse in map_sweeper_to_copy:
            if sweeper.parameter not in [Parameter.attenuation, Parameter.gain, Parameter.current]:
                setattr(map_sweeper_to_copy[pulse], sweeper.parameter.name, original_value[pulse])
