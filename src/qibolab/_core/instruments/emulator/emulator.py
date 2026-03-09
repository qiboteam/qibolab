"""Emulator controller."""

import json
import os
from collections import defaultdict
from collections.abc import Iterable
from functools import reduce
from operator import or_
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator

from qibolab._core.components import Config
from qibolab._core.components.configs import AcquisitionConfig
from qibolab._core.execution_parameters import AveragingMode, ExecutionParameters
from qibolab._core.identifier import Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.pulses import (
    Delay,
    Pulse,
    PulseLike,
    VirtualZ,
)
from qibolab._core.instruments.emulator.hamiltonians import DriveEmulatorConfig, FluxEmulatorConfig
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers

from .engine import (
    Operator,
    OperatorEvolution,
    QutipEngine,
    SimulationEngine,
    TimeDependentOperator,
)
from .engine.abstract import (
    HAMILTONIAN_FILENAME,
    SIMULATOR_CONFIG,
    SWEEP_SIMULATION_FILENAME,
)
from .hamiltonians import (
    HamiltonianConfig,
    Modulated,
    waveform,
)
from .results import acquisitions, index, results

NYQUIST_FREQUENCY = 20
"""GHz, Nyquist frequency used for computing the solution and resolve qubit oscillations."""
SAMPLING_INTERVAL = 1 / (2 * NYQUIST_FREQUENCY)
"""Minimum time the emulator can resolve"""



__all__ = ["EmulatorController"]


class EmulatorController(Controller):
    """Emulator controller."""

    sampling_rate_: float = 1
    """Sampling rate used during simulation."""
    engine: SimulationEngine = QutipEngine()
    """SimulationEngine. Default is QutipEngine."""
    save_dir: os.PathLike | str | None = None
    """Flag for saving the full system evolution computed from the simulation
    backend. In order to set it True modify `platform.py` file in the platform folder."""

    @model_validator(mode="after")
    def validate_save_dir(self):
        if self.save_dir is not None:
            # converting every possible output as a pathlib.Path object
            save_dir = Path(self.save_dir)
            if save_dir.exists():
                raise FileExistsError("The given data folder already exists.")
            object.__setattr__(self, "save_dir", save_dir)
        return self

    @property
    def sampling_rate(self) -> float:
        return self.sampling_rate_

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> float:
        self.sampling_rate_ = value

    def connect(self):
        """Dummy connect method."""

    def disconnect(self):
        """Dummy disconnect method."""

    def _dump_simulation(
        self,
        sequence_idx,
        static_ham: Operator,
        evolution: OperatorEvolution,
        states: NDArray,
        simulation_config: dict,
    ) -> None:
        """Write operators (once), time coefficients (n-d), density matrices (n-d)."""

        if self.save_dir is None:
            return

        sequence_dir = self.save_dir / f"sequence_{sequence_idx}"
        sequence_dir.mkdir(parents=True, exist_ok=True)

        # list of file coefficients; NOTE: the first element is always the simulation timesteps array
        time_coefficients: list[NDArray] = np.stack(
            [evolution.times] + [c for _, c in evolution.operators]
        )

        # solver configuration file path
        json_filename = sequence_dir / (SIMULATOR_CONFIG + ".json")
        static_hamiltonian_filename = sequence_dir / (HAMILTONIAN_FILENAME + ".npy")

        # check if the the sweeper-independent data for the current sequence have already been dumped
        if not static_hamiltonian_filename.exists():
            # list of file operators of the pulse sequence; NOTE: the first element is always the time independent hamiltonian
            operators = np.stack(
                [static_ham.full()] + [op.full() for op, _ in evolution.operators]
            )
            np.save(static_hamiltonian_filename, operators)

        if not json_filename.exists():
            with open(json_filename, "w") as f:
                json.dump(simulation_config, f)

        # NOTE: this might crash if the process is parallelized, to be review once we enable parallelization
        sweep_idx = sum(
            1
            for file in sequence_dir.iterdir()
            if file.is_file() and SWEEP_SIMULATION_FILENAME in file.name
        )
        np.savez(
            sequence_dir / (SWEEP_SIMULATION_FILENAME + f"_{sweep_idx}.npz"),
            time_coeffs=time_coefficients,
            results=states,
            sim_config=simulation_config,
        )

    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[int, Result]:

        if (
            options.averaging_mode is AveragingMode.SINGLESHOT
            and options.nshots is None
        ):
            raise ValueError("nshots must be specified for SINGLESHOT mode")

        # convert align to delays
        sequences_ = (seq.align_to_delays() for seq in sequences)

        results_to_process = (
            self._play_sequence(configs, sequence, options, sweepers)
            for sequence in enumerate(sequences_)
        )

        return reduce(or_, results_to_process)

    def _play_sequence(
        self,
        configs: dict[str, Config],
        sequence: tuple[int, PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ):
        """
        Generate results from an emulated quantum sequence execution.
        Executes a sweep of the quantum sequence and processes the results
        into a structured results object containing quantum states and measurement data.
        """
        sweep_states = self._sweep(sequence, configs, sweepers)
        hamiltonian = cast(HamiltonianConfig, configs["hamiltonian"])
        return results(
            # states in computational basis
            states=sweep_states,
            sequence=sequence[1],
            hamiltonian=hamiltonian,
            options=options,
        )

    def _sweep(
        self,
        sequence: tuple[int, PulseSequence],
        configs: dict[str, Config],
        sweepers: list[ParallelSweepers],
        updates: dict | None = None,
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
            return self._evolve(sequence, configs, updates)

        state_slices: list[NDArray] = []
        parsweep = sweepers[0]
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
            state_slices.append(self._sweep(sequence, configs, sweepers[1:], updates))

        # stack all slices in a single array, along the current outermost dimension
        return np.stack(state_slices)

    def _evolve(
        self,
        sequence_tuple: tuple[int, PulseSequence],
        configs: dict[str, Config],
        updates: dict,
    ) -> NDArray:
        """Evolve a pulse sequence on the quantum emulator.

        This method updates the sequence parameters, generates the time grid, constructs
        the time-dependent Hamiltonian, evolves the initial state with optional collapse
        operators, and returns the resulting measurement data.
        """
        sequence_identifier, sequence = sequence_tuple

        sequence_ = update_sequence(sequence, updates)
        configs_ = update_configs(configs, updates)
        config = cast(HamiltonianConfig, configs_["hamiltonian"])
        hamiltonian = config.hamiltonian(config=configs_, engine=self.engine)
        time_hamiltonian = self._pulse_hamiltonian(sequence_, configs_)
        measurement_times = np.array(
            list(acquisitions(sequence_).values()), dtype=float
        )
        measurement_times[measurement_times < SAMPLING_INTERVAL] = SAMPLING_INTERVAL
        tlist_, index = np.unique(measurement_times, return_inverse=True)

        results, simulation_configs = self.engine.evolve(
            hamiltonian=hamiltonian,
            initial_state=config.initial_state(self.engine),
            time=np.concatenate(([0], tlist_)),
            collapse_operators=config.dissipation(self.engine),
            time_hamiltonian=time_hamiltonian,
        )
        states = np.stack([s.full() for s in results.states[1:]])[index]

        self._dump_simulation(
            sequence_identifier,
            hamiltonian,
            time_hamiltonian,
            states,
            simulation_configs,
        )
        return states

    def _pulse_hamiltonian(
        self, sequence: PulseSequence, configs: dict[str, Config]
    ) -> OperatorEvolution:
        """Construct Hamiltonian time dependent term for qutip simulation."""

        # processed sampling rate; field `sampling_rate` of the `EmulatorController`
        # mimic a real hardware sampling rate, but it is insufficient for us to resolve
        # the oscillation and correctly solve the system evolution, hence we
        # set a nyquist frequency to define the timesteps in order to compute the solution
        times = tlist(sequence)
        channels: list[TimeDependentOperator] = [
            TimeDependentOperator(
                (
                    operator,
                    channel_coefficients(
                        waveforms, sampling_rate=self.sampling_rate, times=times
                    ),
                )
            )
            for operator, waveforms in hamiltonians(
                sequence, configs, self.engine, self.sampling_rate
            )
        ]

        return OperatorEvolution(operators=channels, times=times)

def update_sequence(sequence: PulseSequence, updates: dict) -> PulseSequence:
    """Apply sweep updates to base sequence."""
    return PulseSequence(
        [(ch, e.model_copy(update=updates.get(e.id, {}))) for ch, e in sequence]
    )


def update_configs(configs: dict[str, Config], updates: dict) -> dict[str, Config]:
    """Apply sweep updates to base configs."""
    return {k: c.model_copy(update=updates.get(k, {})) for k, c in configs.items()}


def tlist(sequence: PulseSequence) -> NDArray:
    """Generate a time array for pulse sequence execution.

    This function creates a time array spanning from 0 to the end of the pulse sequence,
    sampled at the Nyquist frequency. If the last element of the sequence is an Acquisition
    or Readout operation, it is excluded from the duration calculation.
    """

    # TODO: maybe this can be a fragility in the case of 0 duration pulses.
    end = max(sequence.duration, SAMPLING_INTERVAL)
    return np.arange(0, end, SAMPLING_INTERVAL)


def hamiltonian(
    pulses: Iterable[PulseLike],
    config: Config,
    hamiltonian: HamiltonianConfig,
    hilbert_space_index: int,
    engine: SimulationEngine,
    sampling_rate: float,
) -> tuple[Operator, list[Modulated]]:
    
    ham_qubit = hamiltonian.qubits[hilbert_space_index]
    n = ham_qubit.transmon_levels
    if isinstance(config, (DriveEmulatorConfig, FluxEmulatorConfig)):
        op = sum((
            engine.expand(
                op=o, 
                targets=hamiltonian.dims, 
                dim=hamiltonian.hilbert_space_index(int(q))
            ) 
            for (q, o) in config.operator(n=n, cross_dict=ham_qubit.crosstalk, engine=engine))
        )
    else:
        op = engine.expand(
            op=config.operator(n=n, engine=engine),
            targets=hilbert_space_index,
            dims=hamiltonian.dims,
        )
    waveforms = (
        waveform(pulse, config, ham_qubit, sampling_rate)
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
    
    hamiltonians_array = ()
    for ch in sequence.channels:
        # TODO: drop the following, and treat acquisitions just as empty channels
        if not isinstance(configs[ch], AcquisitionConfig):
            
            new_terms = hamiltonian(
                            sequence.channel(ch),
                            configs[ch],
                            hconfig,
                            index(ch, hconfig),
                            engine,
                            sampling_rate,
                        )
            hamiltonians_array += (new_terms, )
    return hamiltonians_array


def channel_coefficients(
    waveforms: Iterable[Modulated],
    sampling_rate: int,
    times: NDArray,
) -> NDArray:
    """
    Generate a B-spline interpolation of waveforms over a time evolution.
    This function processes a sequence of pulses, accumulating their waveforms
    over time and applying phase modulation. The resulting waveform is then interpolated
    into a smooth B-spline curve for time evolution analysis.
    """

    pulse_waveforms = np.zeros_like(times)

    cumulative_phase = 0
    cumulative_time = 0
    for pulse in waveforms:
        times_ = times - cumulative_time  # local times
        # in this mask we take into account finite sampling rate with might mismatch with timestep of the emulator
        # (i.e. float time durations and int sampling_rate)
        pulse_times_idx = (times_ >= 0) & (
            times_ < int(pulse.duration * sampling_rate) / sampling_rate
        )
        times_samples = np.floor(
            (times[pulse_times_idx] - cumulative_time) * sampling_rate
        ).astype(int)

        # in case of virtual operations (such as VirtualZ or in general
        # zero-duration pulses), we apply the phase jump without
        # affecting the waveform
        if times_samples.size != 0:
            pulse_waveforms[pulse_times_idx] = pulse(
                times[pulse_times_idx], times_samples, cumulative_phase
            )

        cumulative_phase += pulse.phase
        cumulative_time += pulse.duration

    return pulse_waveforms
