"""Emulator controller."""

from collections import defaultdict
from collections.abc import Iterable
from functools import reduce
from operator import or_
from typing import Callable, Optional, cast

import numpy as np
from numpy.typing import NDArray

from qibolab._core.components import Config
from qibolab._core.components.configs import AcquisitionConfig
from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.identifier import Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.pulses import (
    Acquisition,
    Delay,
    Pulse,
    PulseLike,
    Readout,
    VirtualZ,
)
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers

from .hamiltonians import (
    HamiltonianConfig,
    Modulated,
    Operator,
    waveform,
)
from .operators import TimeDependentOperator, evolve, expand
from .results import acquisitions, results, select_acquisitions

__all__ = ["EmulatorController"]


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

    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[int, Result]:
        # just merge the results of multiple executions in a single dictionary
        return reduce(
            or_,
            (
                results(
                    # states in computational basis
                    self._sweep(sequence, configs, sweepers),
                    sequence,
                    cast(HamiltonianConfig, configs["hamiltonian"]),
                    options,
                )
                for sequence in sequences
            ),
        )

    def _sweep(
        self,
        sequence: PulseSequence,
        configs: dict[str, Config],
        sweepers: list[ParallelSweepers],
        updates: Optional[dict] = None,
    ) -> NDArray:
        """Sweep over sequence.

        This function invokes itself recursively, adding an array
        dimension at each call as the outermost one. The extra dimension
        corresponds to the values in the first nested sweep (with the
        lowest index, interpreted as the outermost as well).
        """
        # use a default dictionary, merging existing values
        updates = defaultdict(dict) | ({} if updates is None else updates)

        if len(sweepers) == 0:
            return self._play_sequence(sequence, configs, updates)

        parsweep = sweepers[0]
        # collect slices of results, corresponding to the current iteration
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

            # append new slice for the current parallel value
            results.append(self._sweep(sequence, configs, sweepers[1:], updates))

        # stack all slices in a single array, along the current outermost dimension
        return np.stack(results)

    def _play_sequence(
        self, sequence: PulseSequence, configs: dict[str, Config], updates: dict
    ) -> NDArray:
        """Play single sequence on emulator.

        The array returned by this function has a single dimension, over
        the various measurements included in the sequence.
        """
        sequence_ = update_sequence(sequence, updates)
        tlist_ = tlist(sequence_, self.sampling_rate)

        configs_ = update_configs(configs, updates)
        config = cast(HamiltonianConfig, configs_["hamiltonian"])
        hamiltonian = config.hamiltonian
        time_hamiltonian = self._pulse_hamiltonian(sequence_, configs_)
        if time_hamiltonian is not None:
            hamiltonian += time_hamiltonian
        results = evolve(
            hamiltonian,
            config.initial_state,
            tlist_,
            config.dissipation,
        )
        return select_acquisitions(
            results.states,
            acquisitions(sequence_).values(),
            tlist_,
        )

    def _pulse_hamiltonian(
        self, sequence: PulseSequence, configs: dict[str, Config]
    ) -> Optional[TimeDependentOperator]:
        """Construct Hamiltonian time dependent term for qutip simulation."""

        channels = [
            [operator, channel_time(waveforms)]
            for operator, waveforms in hamiltonians(sequence, configs)
        ]
        return TimeDependentOperator(channels) if len(channels) > 0 else None


def update_sequence(sequence: PulseSequence, updates: dict) -> PulseSequence:
    """Apply sweep updates to base sequence."""
    return PulseSequence(
        [(ch, e.model_copy(update=updates.get(e.id, {}))) for ch, e in sequence]
    )


def update_configs(configs: dict[str, Config], updates: dict) -> dict[str, Config]:
    """Apply sweep updates to base configs."""
    return {k: c.model_copy(update=updates.get(k, {})) for k, c in configs.items()}


def tlist(
    sequence: PulseSequence, sampling_rate: float, per_sample: float = 2
) -> NDArray:
    """Compute times for evolution.

    The frequency of times is double the sampling rate, to make sure
    that all pulses features are resolved by the evolution.

    This can be customized using the `per_sample` rate, e.g. to retrieve times at the
    sampling rate itself, for pulses evaluation.

    .. note::

        As an optimization, if an acquisition is executed as the last
        sequence operation, that's not taken into account, since it is not
        simulated by the present emulator.

        For long experiments, it is a mild optimization. But it critically speeds up
        short experiments, given the usual relative duration of acquisitions and control
        pulses.
    """
    seq = (
        sequence[:-1]
        if isinstance(sequence[-1][1], (Acquisition, Readout))
        else sequence
    )
    end = max(seq.duration, 1)
    rate = sampling_rate * per_sample
    return np.arange(0, end, 1 / rate)


def hamiltonian(
    pulses: Iterable[PulseLike],
    config: Config,
    hamiltonian: HamiltonianConfig,
    qubit: int,
) -> tuple[Operator, list[Modulated]]:
    n = hamiltonian.transmon_levels
    op = expand(config.operator(n), hamiltonian.dims, qubit)
    waveforms = (
        waveform(pulse, config)
        for pulse in pulses
        # only handle pulses (thus no readout)
        if isinstance(pulse, (Pulse, Delay, VirtualZ))
    )
    return (op, [w for w in waveforms if w is not None])


def hamiltonians(
    sequence: PulseSequence, configs: dict[str, Config]
) -> Iterable[tuple[Operator, list[Modulated]]]:
    hconfig = cast(HamiltonianConfig, configs["hamiltonian"])
    # TODO: pass qubit in a better way
    return (
        hamiltonian(sequence.channel(ch), configs[ch], hconfig, int(ch[0]))
        for ch in sequence.channels
        # TODO: drop the following, and treat acquisitions just as empty channels
        if not isinstance(configs[ch], AcquisitionConfig)
    )


def channel_time(waveforms: Iterable[Modulated]) -> Callable[[float], float]:
    """Wrap time function for specific channel.

    Used to avoid late binding issues.
    """

    def time(t: float) -> float:
        cumulative_time = 0
        cumulative_phase = 0
        for pulse in waveforms:
            pulse_duration = pulse.duration  # TODO: pass sampling rate
            pulse_phase = pulse.phase
            if cumulative_time <= t < cumulative_time + pulse_duration:
                relative_time = t - cumulative_time
                index = int(relative_time)  # TODO: pass sampling rate
                return pulse(t, index, cumulative_phase)
            cumulative_time += pulse_duration
            cumulative_phase += pulse_phase
        return 0

    return time
