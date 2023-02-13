import copy
import time

import numpy as np
from qibo.config import log, raise_error

from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence, PulseType
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

    def sweep(self, sequence, *sweepers, nshots=1024, average=True, relaxation_time=None):
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
            relaxation_time=relaxation_time,
            results=results,
            sweeper_pulses=sweeper_pulses,
            map_original_shifted=map_original_shifted
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
        map_original_shifted=None
    ):
        sweeper = sweepers[0]

        # store values before starting to sweep
        if sweeper.pulses is not None:
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
                    map_original_shifted=map_original_shifted
                )
            else:
                new_sequence = copy.deepcopy(sequence)
                result = self.execute_pulse_sequence(new_sequence, nshots)
                # colllect result and append to original pulse
                for original_pulse, new_serial in map_original_shifted.items():
                    acquisition = result[new_serial].compute_average() if average else result[new_serial]

                    if results:
                        results[original_pulse.serial] += acquisition
                        results[original_pulse.qubit] += acquisition
                    else:
                        results[original_pulse.serial] = acquisition
                        results[original_pulse.qubit] = copy.copy(results[original_pulse.serial])

        # restore initial value of the pulse
        if sweeper.pulses is not None:
            self._restore_initial_value(sweeper, sweeper_pulses, original_value)

    def _save_original_value(self, sweeper, sweeper_pulses):
        """Helper method for _sweep_recursion"""
        original_value = {}
        pulses = sweeper_pulses[sweeper.parameter]
        # save original value of the parameter swept
        for pulse in pulses:
            if sweeper.parameter not in [Parameter.attenuation, Parameter.gain, Parameter.bias]:
                original_value[pulse] = getattr(pulses[pulse], sweeper.parameter.name)

        return original_value

    def _restore_initial_value(self, sweeper, sweeper_pulses, original_value):
        """Helper method for _sweep_recursion"""
        pulses = sweeper_pulses[sweeper.parameter]
        for pulse in pulses:
            if sweeper.parameter not in [Parameter.attenuation, Parameter.gain, Parameter.bias]:
                setattr(pulses[pulse], sweeper.parameter.name, original_value[pulse])

    def _update_pulse_sequence_parameters(
        self, sweeper, sweeper_pulses, original_sequence, map_original_shifted, value
    ):
        """Helper method for _sweep_recursion"""
        if sweeper.pulses is not None:
            pulses = sweeper_pulses[sweeper.parameter]
            for pulse in pulses:
                if sweeper.parameter is Parameter.frequency:
                    if pulses[pulse].type is PulseType.READOUT:
                        value += self.qubits[pulses[pulse].qubit].readout_frequency
                    else:
                        value += self.qubits[pulses[pulse].qubit].drive_frequency
                    setattr(pulses[pulse], sweeper.parameter.name, value)
                elif sweeper.parameter is Parameter.amplitude:
                    if pulses[pulse].type is PulseType.READOUT:
                        current_amplitude = self.native_gates["single_qubit"][pulses[pulse].qubit]["MZ"]["amplitude"]
                    else:
                        current_amplitude = self.native_gates["single_qubit"][pulses[pulse].qubit]["RX"]["amplitude"]
                    setattr(pulses[pulse], sweeper.parameter.name, float(current_amplitude * value))
                else:
                    setattr(pulses[pulse], sweeper.parameter.name, value)
                if pulses[pulse].type is PulseType.READOUT:
                    to_modify = [
                        pulse1 for pulse1 in original_sequence.ro_pulses if pulse1.qubit == pulses[pulse].qubit
                    ]
                    if to_modify:
                        map_original_shifted[to_modify[0]] = pulses[pulse].serial

        if sweeper.qubits is not None:
            for qubit in sweeper.qubits:
                if sweeper.parameter is Parameter.attenuation:
                    self.set_attenuation(qubit, value)
                elif sweeper.parameter is Parameter.gain:
                    self.set_gain(qubit, value)
                elif sweeper.parameter is Parameter.bias:
                    self.set_bias(qubit, value)
