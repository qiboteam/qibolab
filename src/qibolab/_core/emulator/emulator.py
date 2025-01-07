"""Emulator controller."""

import numpy as np
from qibo.config import log
from qutip import basis, mesolve, tensor

from qibolab import Readout
from qibolab._core.components import AcquisitionConfig, Config
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.instruments.abstract import Controller
from qibolab._core.sequence import PulseSequence

from .hamiltonians import waveform
from .utils import merge_results


class EmulatorController(Controller):
    """Emulator controller."""

    bounds: str = "emulator/bounds"

    def connect(self):
        log.info("Starting emulator.")

    def disconnect(self):
        log.info("Stopping emulator.")

    @property
    def sampling_rate(self):
        """Sampling rate of emulator."""
        # TODO: remove hardcoded sampling rate
        return 1

    @property
    def initial_state(self):
        """System in ground state."""
        # initial state: qubit in ground state
        return tensor(basis(3, 0))

    def probability(self, state: int):
        """Probability of qubit in state."""
        return basis(3, state) * basis(3, state).dag()

    def _play_sequence(self, sequence, configs, options, updates=None):
        """Play single sequence on emulator."""

        if updates is None:
            updates = {}

        hamiltonian = configs["hamiltonian"].hamiltonian
        hamiltonian += self._pulse_sequence_to_hamiltonian(sequence, configs, updates)
        measurement, tlist = self._measurement(sequence, configs, updates)
        results = mesolve(
            hamiltonian,
            self.initial_state,
            tlist,
            configs["hamiltonian"].decoherence,
            [self.probability(0), self.probability(1)],
        )
        averaged_results = {
            ro_pulse_id: [results.expect[0][sample - 1], results.expect[1][sample - 1]]
            for ro_pulse_id, sample in measurement.items()
        }

        if options.averaging_mode == AveragingMode.SINGLESHOT:
            results = {
                ro_pulse_id: np.random.choice(
                    [0, 1, 2],
                    size=options.nshots,
                    p=[max(prob[0], 0), max(prob[1], 0), max(0, 1 - prob[0] - prob[1])],
                )
                for ro_pulse_id, prob in averaged_results.items()
            }
            return results

        # dropping probability of 0 to keep compatibility with qibolab
        return {pulse: prob[1] for pulse, prob in averaged_results.items()}

    def _sweep(self, sequence, configs, options, sweepers, updates=None):
        """Sweep over sequence."""
        if updates is None:
            updates = {}
        results = {}
        assert len(sweepers[0]) == 1, "Parallel sweepers not supported."
        sweeper = sweepers[0][0]
        for value in sweeper.values:
            if sweeper.pulses is not None:
                for pulse in sweeper.pulses:
                    try:
                        updates[pulse.id].update({sweeper.parameter.name: value})
                    except KeyError:
                        updates[pulse.id] = {sweeper.parameter.name: value}
            if sweeper.channels is not None:
                for channel in sweeper.channels:
                    try:
                        updates[channel].update({sweeper.parameter.name: value})
                    except KeyError:
                        updates[channel] = {sweeper.parameter.name: value}
            if len(sweepers) > 1:
                temp = self._sweep(sequence, configs, options, sweepers[1:], updates)
            else:
                temp = self._play_sequence(sequence, configs, options, updates)

            results = merge_results(
                results,
                temp,
            )

        # reshaping results
        for key, value in results.items():
            results[key] = results[key].reshape(options.results_shape(sweepers))
        return results

    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list = None,
    ):
        assert (
            options.acquisition_type == AcquisitionType.DISCRIMINATION
        ), "Emulator only supports DISCRIMINATION acquisition type."
        results = {}

        for sequence in sequences:
            if len(sweepers) > 0:
                results.update(self._sweep(sequence, configs, options, sweepers))
            else:
                results.update(self._play_sequence(sequence, configs, options))
        return results

    def _measurement(self, sequence, configs, updates=None):
        """Given sequence creates a dictionary of readout pulses and their
        sample index."""
        duration = 0
        pulses = {}
        if updates is None:
            updates = {}

        for channel, pulse in sequence:
            if isinstance(configs[channel], AcquisitionConfig):
                if isinstance(pulse, Readout):
                    pulses[pulse.id] = int(duration)
                if pulse.id in updates:
                    pulse = pulse.model_copy(update=updates[pulse.id])
                duration += pulse.duration

        tmax = int(max(pulses.values()) * self.sampling_rate)
        if tmax > 0:
            # TODO: less steps to speed up simulation
            tlist = np.arange(0, tmax)
        else:
            tlist = np.arange(0, 1)
        return pulses, tlist

    def _pulse_sequence_to_hamiltonian(
        self, sequence: PulseSequence, configs, updates=None
    ) -> dict[str, list]:
        """Construct Hamiltonian dependent term for qutip simulation."""

        hamiltonians = {}
        h_t = []

        for channel, pulse in sequence:
            # do not handle readout pulses
            if not isinstance(configs[channel], AcquisitionConfig):
                signal = waveform(pulse, channel, configs, updates)
                if channel in hamiltonians:
                    hamiltonians[channel] += [signal]
                else:
                    hamiltonians[channel] = [signal]

        for channel, waveforms in hamiltonians.items():

            def time(t, args=None):
                cumulative_time = 0
                for pulse in waveforms:
                    pulse_duration = len(pulse) * 1  # TODO: pass sampling rate
                    if cumulative_time <= t < cumulative_time + pulse_duration:
                        relative_time = t - cumulative_time
                        index = int(relative_time // 1)  # TODO: pass sampling rate
                        return pulse(t, index)
                    cumulative_time += pulse_duration
                return 0

            h_t.append([waveforms[0].operator, time])
        return h_t
