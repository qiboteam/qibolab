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

from .engine import Operator, OperatorEvolution, QutipEngine, SimulationEngine
from .hamiltonians import (
    HamiltonianConfig,
    Modulated,
    waveform,
)
from .results import acquisitions, index, results, select_acquisitions

__all__ = ["EmulatorController"]


class EmulatorController(Controller):
    """Emulator controller."""

    sampling_rate_: float = 1
    """Sampling rate used during simulation."""
    engine: SimulationEngine = QutipEngine()
    """SimulationEngine. Default is QutipEngine."""
    bounds: str = "emulator/bounds"
    """Bounds for emulator."""

    @property
    def sampling_rate(self) -> float:
        return self.sampling_rate_

    @sampling_rate.setter
    def sampling_rate(self, value: float):
        self.sampling_rate_ = value

    def connect(self):
        """Dummy connect method."""

    def disconnect(self):
        """Dummy disconnect method."""

    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[int, Result]:
        # convert align to delays
        sequences_ = (seq.align_to_delays() for seq in sequences)

        results_to_process = (
            self._results(configs, sequence, options, sweepers)
            for sequence in sequences_
        )

        return reduce(or_, results_to_process)

    def _results(
        self,
        configs: dict[str, Config],
        sequence: PulseSequence,
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ):
        """
        Generate results from an emulated quantum sequence execution.
        Executes a sweep of the quantum sequence and processes the results
        into a structured results object containing quantum states and measurement data.
        """

        sweep_results = self._sweep(sequence, configs, sweepers)
        hamiltonian = cast(HamiltonianConfig, configs["hamiltonian"])
        return results(
            # states in computational basis
            states=sweep_results,
            sequence=sequence,
            hamiltonian=hamiltonian,
            options=options,
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
        """Execute a pulse sequence on the quantum emulator.

        This method updates the sequence parameters, generates the time grid, constructs
        the time-dependent Hamiltonian, evolves the initial state with optional collapse
        operators, and returns the resulting measurement data.
        """
        sequence_ = update_sequence(sequence, updates)
        tlist_ = tlist(sequence_, self.sampling_rate, per_sample=2)
        configs_ = update_configs(configs, updates)
        config = cast(HamiltonianConfig, configs_["hamiltonian"])
        hamiltonian = config.hamiltonian(config=configs_, engine=self.engine)
        time_hamiltonian = self._pulse_hamiltonian(sequence_, configs_)
        results = self.engine.evolve(
            hamiltonian=hamiltonian,
            initial_state=config.initial_state(self.engine),
            time=tlist_,
            collapse_operators=config.dissipation(self.engine),
            time_hamiltonian=time_hamiltonian,
        )
        return select_acquisitions(
            results.states,
            acquisitions(sequence_).values(),
            tlist_,
        )

    def _pulse_hamiltonian(
        self, sequence: PulseSequence, configs: dict[str, Config]
    ) -> Optional[OperatorEvolution]:
        """Construct Hamiltonian time dependent term for qutip simulation."""

        channels = [
            [
                operator,
                channel_time(waveforms, sampling_rate=self.sampling_rate),
            ]
            for operator, waveforms in hamiltonians(
                sequence, configs, self.engine, self.sampling_rate
            )
        ]
        return OperatorEvolution(channels) if len(channels) > 0 else None


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
    hilbert_space_index: int,
    engine: SimulationEngine,
    sampling_rate: float,
) -> tuple[Operator, list[Modulated]]:
    n = hamiltonian.transmon_levels
    op = engine.expand(
        config.operator(n=n, engine=engine), hamiltonian.dims, hilbert_space_index
    )
    waveforms = (
        waveform(pulse, config, hamiltonian.qubits[hilbert_space_index], sampling_rate)
        for pulse in pulses
        if isinstance(pulse, (Pulse, Delay, VirtualZ))
    )
    return (op, [w for w in waveforms if w is not None])


def hamiltonians(
    sequence: PulseSequence,
    configs: dict[str, Config],
    engine: SimulationEngine,
    sampling_rate: float,
) -> Iterable[tuple[Operator, list[Modulated]]]:
    hconfig = cast(HamiltonianConfig, configs["hamiltonian"])
    return (
        hamiltonian(
            sequence.channel(ch),
            configs[ch],
            hconfig,
            index(ch, hconfig),
            engine,
            sampling_rate,
        )
        for ch in sequence.channels
        # TODO: drop the following, and treat acquisitions just as empty channels
        if not isinstance(configs[ch], AcquisitionConfig)
    )


def channel_time(
    waveforms: Iterable[Modulated],
    sampling_rate: int,
) -> Callable[[float], float]:
    """Wrap time function for specific channel.

    Used to avoid late binding issues.
    """

    def time(t: float) -> float:
        cumulative_time = 0
        cumulative_phase = 0
        for pulse in waveforms:
            pulse_phase = pulse.phase
            if cumulative_time <= t < cumulative_time + pulse.duration:
                relative_time = t - cumulative_time
                index = int(np.floor(relative_time * sampling_rate))
                return pulse(t, index, cumulative_phase)
            cumulative_time += pulse.duration
            cumulative_phase += pulse_phase
        return 0

    return time
