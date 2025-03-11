"""Emulator controller."""

from collections import defaultdict
from functools import reduce
from operator import or_
from typing import Any, Optional, cast

import numpy as np
from numpy.typing import NDArray
from qutip import mesolve

from qibolab import Readout
from qibolab._core.components import AcquisitionConfig, Config
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.pulses.pulse import PulseId
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers

from .hamiltonians import HamiltonianConfig, waveform
from .operators import INITIAL_STATE, SIGMAZ
from .utils import ndchoice


class EmulatorController(Controller):
    """Emulator controller."""

    bounds: str = "emulator/bounds"

    def connect(self):
        """Dummy connect method."""

    def disconnect(self):
        """Dummy disconnect method."""

    @property
    def sampling_rate(self):
        """Sampling rate of emulator."""
        return 1

    @property
    def initial_state(self):
        """System in ground state."""
        # initial state: qubit in ground state
        return INITIAL_STATE

    def _play_sequence(
        self, sequence: PulseSequence, configs: dict[str, Config], updates: dict
    ) -> NDArray:
        """Play single sequence on emulator."""
        config = cast(HamiltonianConfig, configs["hamiltonian"])
        hamiltonian = config.hamiltonian
        hamiltonian += self._pulse_sequence_to_hamiltonian(sequence, configs, updates)
        measurements, tlist = self._measurement(sequence, configs, updates)
        results = mesolve(
            hamiltonian, self.initial_state, tlist, config.decoherence, e_ops=[SIGMAZ]
        )
        samples = np.array(list(measurements.values())) - 1
        return (1 - results.expect[0][samples]) / 2

    def _sweep(
        self,
        sequence: PulseSequence,
        configs: dict[str, Config],
        sweepers: list[ParallelSweepers],
        updates: Optional[dict] = None,
    ) -> NDArray:
        """Sweep over sequence."""
        updates = defaultdict(dict) | ({} if updates is None else updates)
        if len(sweepers) == 0:
            return self._play_sequence(sequence, configs, updates)

        parsweep = sweepers[0]
        results = []
        # execute once for each parallel value
        for values in zip(*(s.values for s in parsweep)):
            # update all parallel sweepers with the respective values
            for sweeper, value in zip(parsweep, values):
                if sweeper.pulses is not None:
                    for pulse in sweeper.pulses:
                        updates[pulse.id].update({sweeper.parameter.name: value})
                if sweeper.channels is not None:
                    for channel in sweeper.channels:
                        updates[channel].update({sweeper.parameter.name: value})
            results.append(self._sweep(sequence, configs, sweepers[1:], updates))

        return np.vstack(results)

    def _play(
        self,
        sequence: PulseSequence,
        configs: dict[str, Config],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[int, Result]:
        res = self._sweep(sequence, configs, sweepers)
        res_ = (
            np.moveaxis(
                ndchoice(np.moveaxis(np.array([1 - res, res]), 0, -1), options.nshots),
                -1,
                0,
            )
            if options.averaging_mode == AveragingMode.SINGLESHOT
            else res
        )
        # move measurements dimension to the front, and reshape
        res__ = np.moveaxis(res_, -1, 0).reshape(
            res_.shape[-1], *options.results_shape(sweepers)
        )
        measurements, _ = self._measurement(sequence, configs, {})
        return dict(zip(measurements.keys(), list(res__)))

    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[int, Result]:
        assert options.acquisition_type == AcquisitionType.DISCRIMINATION, (
            "Emulator only supports DISCRIMINATION acquisition type."
        )
        return reduce(
            or_,
            (
                self._play(sequence, configs, options, sweepers)
                for sequence in sequences
            ),
        )

    def _measurement(
        self,
        sequence: PulseSequence,
        configs: dict[str, Config],
        updates: dict,
    ) -> tuple[dict[PulseId, Any], NDArray]:
        """Given sequence creates a dictionary of readout pulses and their
        sample index."""
        duration = 0
        pulses = {}
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
        self, sequence: PulseSequence, configs: dict[str, Config], updates: dict
    ) -> dict[str, list]:
        """Construct Hamiltonian dependent term for qutip simulation."""

        hamiltonians = defaultdict(list)
        h_t = []
        for channel, pulse in sequence:
            # do not handle readout pulses
            if not isinstance(configs[channel], AcquisitionConfig):
                signal = waveform(pulse, channel, configs, updates)
                hamiltonians[channel] += [signal]

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
