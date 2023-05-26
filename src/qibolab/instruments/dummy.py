import copy
import time
from typing import Dict, List, Union

import numpy as np
from qibo.config import log, raise_error

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.instruments.abstract import AbstractInstrument
from qibolab.platforms.abstract import Qubit
from qibolab.pulses import PulseSequence, PulseType
from qibolab.sweeper import Parameter, Sweeper


class DummyInstrument(AbstractInstrument):
    """Dummy instrument that returns random voltage values.

    Useful for testing code without requiring access to hardware.

    Args:
        name (str): name of the instrument.
        address (int): address to connect to the instrument.
            Not used since the instrument is dummy, it only
            exists to keep the same interface with other
            instruments.
    """

    def connect(self):
        log.info("Connecting to dummy instrument.")

    def setup(self, *args, **kwargs):
        log.info("Setting up dummy instrument.")

    def start(self):
        log.info("Starting dummy instrument.")

    def stop(self):
        log.info("Stopping dummy instrument.")

    def disconnect(self):
        log.info("Disconnecting dummy instrument.")

    def play(self, qubits: Dict[Union[str, int], Qubit], sequence: PulseSequence, options: ExecutionParameters):
        nshots = options.nshots
        time.sleep(options.relaxation_time * 1e-9)

        ro_pulses = {pulse.qubit: pulse.serial for pulse in sequence.ro_pulses}

        results = {}
        for qubit, serial in ro_pulses.items():
            if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                samples = (
                    np.random.rand(1) if options.averaging_mode is AveragingMode.CYCLIC else np.random.rand(nshots)
                )
                results[qubit] = results[serial] = options.results_type(samples)

            else:
                i = np.random.rand(1) if options.averaging_mode is AveragingMode.CYCLIC else np.random.rand(nshots)
                q = np.random.rand(1) if options.averaging_mode is AveragingMode.CYCLIC else np.random.rand(nshots)
                exp_res = i + 1j * q
                results[qubit] = results[serial] = options.results_type(data=exp_res)

            # results[qubit] = results[serial] = ExecutionResults.from_components(i, q, shots)

        return results

    def sweep(
        self,
        qubits: Dict[Union[str, int], Qubit],
        sequence: PulseSequence,
        options: ExecutionParameters,
        *sweepers: List[Sweeper]
    ):
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
            qubits,
            copy_sequence,
            copy.deepcopy(sequence),
            options,
            *sweepers,
            results=results,
            sweeper_pulses=sweeper_pulses,
            map_original_shifted=map_original_shifted
        )
        return results

    def _sweep_recursion(
        self,
        qubits,
        sequence,
        original_sequence,
        options,
        *sweepers,
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
                qubits, sweeper, sweeper_pulses, original_sequence, map_original_shifted, value
            )
            if len(sweepers) > 1:
                self._sweep_recursion(
                    qubits,
                    sequence,
                    original_sequence,
                    options,
                    *sweepers[1:],
                    results=results,
                    sweeper_pulses=sweeper_pulses,
                    map_original_shifted=map_original_shifted
                )
            else:
                new_sequence = copy.deepcopy(sequence)
                result = self.play(qubits, new_sequence, options)

                # colllect result and append to original pulse
                for original_pulse, new_serial in map_original_shifted.items():
                    acquisition = result[new_serial]
                    if original_pulse.serial in results:
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
            if sweeper.parameter not in [Parameter.attenuation, Parameter.gain, Parameter.bias, Parameter.delay]:
                original_value[pulse] = getattr(pulses[pulse], sweeper.parameter.name)
        return original_value

    def _restore_initial_value(self, sweeper, sweeper_pulses, original_value):
        """Helper method for _sweep_recursion"""
        pulses = sweeper_pulses[sweeper.parameter]
        for pulse in pulses:
            if sweeper.parameter not in [Parameter.attenuation, Parameter.gain, Parameter.bias, Parameter.delay]:
                setattr(pulses[pulse], sweeper.parameter.name, original_value[pulse])

    def _update_pulse_sequence_parameters(
        self, qubits, sweeper, sweeper_pulses, original_sequence, map_original_shifted, value
    ):
        """Helper method for _sweep_recursion"""
        if sweeper.pulses is not None:
            pulses = sweeper_pulses[sweeper.parameter]
            for pulse in pulses:
                if sweeper.parameter is Parameter.frequency:
                    if pulses[pulse].type is PulseType.READOUT:
                        value += qubits[pulses[pulse].qubit].readout_frequency
                    else:
                        value += qubits[pulses[pulse].qubit].drive_frequency
                    setattr(pulses[pulse], sweeper.parameter.name, value)
                elif sweeper.parameter is Parameter.amplitude:
                    current_amplitude = pulses[pulse].amplitude
                    setattr(pulses[pulse], sweeper.parameter.name, float(current_amplitude * value))
                elif sweeper.parameter is Parameter.delay:
                    pulses[pulse].start += value
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
                    qubit.readout.attenuation = value
                elif sweeper.parameter is Parameter.gain:
                    qubit.drive.gain = value
                elif sweeper.parameter is Parameter.bias:
                    qubit.flux.offset = float(value)
