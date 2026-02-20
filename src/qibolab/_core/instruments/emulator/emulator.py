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
                for sequence in sequences_
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
        tlist_ = tlist(sequence_, self.sampling_rate, per_sample=2)
        configs_ = update_configs(configs, updates)
        config = cast(HamiltonianConfig, configs_["hamiltonian"])
        hamiltonian = config.hamiltonian(config=configs_, engine=self.engine)
        time_hamiltonian = self._pulse_hamiltonian(sequence_, configs_)
        dimensions = config.hilbert_space_dims(self.engine.has_flipped_index)
        results = self.engine.evolve(
            hamiltonian=hamiltonian,
            initial_state=config.initial_state(self.engine),
            time=tlist_,
            collapse_operators=config.dissipation(self.engine),
            time_hamiltonian=time_hamiltonian,
            dimensions=dimensions,
        )
        evolution_states = self.engine.get_evolution_states(results)
        
        return select_acquisitions(
            evolution_states,
            acquisitions(sequence_).values(),
            tlist_,
            engine=self.engine,
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


class CudaqEmulatorController(EmulatorController):
    def _sweep(
        self,
        sequence: PulseSequence,
        configs: dict[str, Config],
        sweepers: list[ParallelSweepers],
        updates: Optional[dict] = None,
    ) -> list:
        """Processes sweepers using _make_sweep_list to reconstruct the corresponding list of hamiltonians for batch execution."""
        
        output_list = self._make_sweep_list(sequence, configs, sweepers, updates)
        
        # retrieve dimensions from default config
        config = cast(HamiltonianConfig, configs["hamiltonian"])
        dimensions = config.hilbert_space_dims(self.engine.has_flipped_index)

        # for non-sweep executions, reshape output_list
        if isinstance(output_list, tuple):
            output_list = [output_list]
        
        hamiltonian_list = list(get_flattened_list(output_list, 0))
        duration_list = list(get_flattened_list(output_list, 2))
        sequence_list = list(get_flattened_list(output_list, 1))
        max_duration_index = int(np.argmax(duration_list))
        max_duration_sequence = sequence_list[max_duration_index]
        tlist_ = tlist(max_duration_sequence, self.sampling_rate, per_sample=2)

        results = self.engine.evolve(
            hamiltonian=hamiltonian_list,
            initial_state=config.initial_state(self.engine),
            time=tlist_,
            collapse_operators=[config.dissipation(self.engine) for _ in hamiltonian_list],
            time_hamiltonian=None,
            dimensions=dimensions,
        )
        
        # for non-sweep executions, reshape output_list
        if not isinstance(results, list):
            results = [results]

        sweep_acquisitions = [
            select_acquisitions(
                self.engine.get_evolution_states(results[i]), 
                acquisitions(sequence).values(), 
                tlist_,
                engine=self.engine,
                statevector_dimension=np.prod(list(dimensions.values())),
            ) 
            for i, sequence in enumerate(sequence_list)]

        # stack all slices in a single array, along the current outermost dimension
        sweep_acquisitions = np.stack(sweep_acquisitions)

        sweepers_shape = []
        for sweeper in sweepers:
            sweepers_shape.append(len(sweeper[0].values))
        sweepers_shape += list(sweep_acquisitions.shape[1:])
        sweep_acquisitions = np.stack(sweep_acquisitions).reshape(sweepers_shape)
        self.sweep_acquisitions = sweep_acquisitions
        
        return sweep_acquisitions

    def _make_sweep_list(
        self,
        sequence: PulseSequence,
        configs: dict[str, Config],
        sweepers: list[ParallelSweepers],
        updates: Optional[dict] = None,
    ) -> list:
        """Adapted method from EmulatorController._sweep to generate list of evolve input tuples from sweepers."""  
        
        # use a default dictionary, merging existing values
        updates = defaultdict(dict) | ({} if updates is None else updates)

        if len(sweepers) == 0:
            sequence_ = update_sequence(sequence, updates)
            tlist_ = tlist(sequence_, self.sampling_rate, per_sample=2)
            configs_ = update_configs(configs, updates)
            config = cast(HamiltonianConfig, configs_["hamiltonian"])
            hamiltonian = config.hamiltonian(config=configs_, engine=self.engine)
            time_hamiltonian = self._pulse_hamiltonian(sequence_, configs_)
            if time_hamiltonian:
                for op, waveform in time_hamiltonian.operators:
                    hamiltonian += self.engine.engine.ScalarOperator(waveform) * op
            
            return (hamiltonian, sequence_, tlist_[-1])

        parsweep = sweepers[0]
        # collect slices of results, corresponding to the current iteration
        output_list = []
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
            current_output = self._make_sweep_list(sequence, configs, sweepers[1:], updates)
            output_list.append(current_output)

        return output_list


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
    pulse_index: int,
    hilbert_space_index: int,
    engine: SimulationEngine,
    sampling_rate: float,
) -> tuple[Operator, list[Modulated]]:
    n = hamiltonian.transmon_levels
    op = engine.expand(
        config.operator(n=n, target=hilbert_space_index, engine=engine), hamiltonian.dims, hilbert_space_index
    )
    waveforms = (
        waveform(pulse, config, hamiltonian.qubits[pulse_index], sampling_rate)
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
            index(ch, hconfig, engine.has_flipped_index),
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
            if cumulative_time <= abs(t) < cumulative_time + pulse.duration:
                relative_time = np.real(t - cumulative_time)
                index = int(np.floor(relative_time * sampling_rate))
                return pulse(t, index, cumulative_phase)
            cumulative_time += pulse.duration
            cumulative_phase += pulse_phase
        return 0

    return time

def get_flattened_list(lst, index):
    # index = 0 for hamiltonian, 1 for sequence, 2 for sequence duration
    for item in lst:
        if isinstance(item, list):
            yield from get_flattened_list(item, index)
        else:
            yield item[index]
