"""Emulator controller."""

from collections import defaultdict
from collections.abc import Iterable
from functools import reduce
from operator import or_
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline, make_interp_spline

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
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers

from .engine import Operator, OperatorEvolution, QutipEngine, SimulationEngine
from .hamiltonians import (
    HamiltonianConfig,
    Modulated,
    waveform,
)
from .results import acquisitions, index, results

SPLINE_INTERP_ORDER = 3
"""Polynomial order used for interpolating the pulses with a spline function."""
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
    save_dir: Path | None = None
    """Flag for saving the full system evolution computed from the simulation
    backend. In order to set it True modify `platform.py` file in the platform folder."""

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

    def _dump_simulation(self, sequence, configs, states, coefficients):
        """Write operators (once), time coefficients (n-d), density matrices (n-d)."""
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # operators: run-invariant, so build them once here (outside the sweep loop)
        config = cast(HamiltonianConfig, configs["hamiltonian"])
        static = config.hamiltonian(config=configs, engine=self.engine)
        evolution = self._pulse_hamiltonian(sequence, configs)
        time_ops = (
            [pair[0] for pair in evolution.operators] if evolution is not None else []
        )
        operators = [static] + time_ops

        self.engine.save_operators(operators, self.save_dir)
        np.save(self.save_dir / "time_coefficients.npy", coefficients)
        np.save(self.save_dir / "density_matrices.npy", states)

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
            for sequence in sequences_
        )

        return reduce(or_, results_to_process)

    def _play_sequence(
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
        sweep_states, sweep_coefficients = self._sweep(sequence, configs, sweepers)
        if self.save_dir is not None:
            self._dump_simulation(sequence, configs, sweep_states, sweep_coefficients)
        hamiltonian = cast(HamiltonianConfig, configs["hamiltonian"])
        return results(
            # states in computational basis
            states=sweep_states,
            sequence=sequence,
            hamiltonian=hamiltonian,
            options=options,
        )

    def _sweep(
        self,
        sequence: PulseSequence,
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

        state_slices, coeff_slices = [], []
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
        states, coeffs = self._sweep(sequence, configs, sweepers[1:], updates)
        state_slices.append(states)
        coeff_slices.append(coeffs)

        # stack all slices in a single array, along the current outermost dimension
        return np.stack(state_slices), np.stack(coeff_slices)

    def _evolve(
        self, sequence: PulseSequence, configs: dict[str, Config], updates: dict
    ) -> NDArray:
        """Evolve a pulse sequence on the quantum emulator.

        This method updates the sequence parameters, generates the time grid, constructs
        the time-dependent Hamiltonian, evolves the initial state with optional collapse
        operators, and returns the resulting measurement data.
        """
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

        results = self.engine.evolve(
            hamiltonian=hamiltonian,
            initial_state=config.initial_state(self.engine),
            time=np.concatenate(([0], tlist_)),
            collapse_operators=config.dissipation(self.engine),
            time_hamiltonian=time_hamiltonian,
        )
        states = np.stack([s.full() for s in results.states[1:]])[index]
        coefficients = (
            time_hamiltonian.coefficients if time_hamiltonian is not None else None
        )
        return states, coefficients

    def _pulse_hamiltonian(
        self, sequence: PulseSequence, configs: dict[str, Config]
    ) -> OperatorEvolution | None:
        """Construct Hamiltonian time dependent term for qutip simulation."""

        # processed sampling rate; field `sampling_rate` of the `EmulatorController`
        # mimic a real hardware sampling rate, but it is insufficient for us to resolve
        # the oscillation and correctly solve the system evolution, hence we
        # set a nyquist frequency to define the timesteps in order to compute the solution
        times = tlist(sequence)
        channels, raw_coefficients = [], []
        for operator, waveforms in hamiltonians(
            sequence, configs, self.engine, self.sampling_rate
        ):
            raw, spline = channel_coefficients(
                waveforms, sampling_rate=self.sampling_rate, times=times
            )
            channels.append([operator, spline])
            raw_coefficients.append(raw)
        if len(channels) == 0:
            return None

        return OperatorEvolution(channels) if len(channels) > 0 else None


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


def channel_coefficients(
    waveforms: Iterable[Modulated],
    sampling_rate: int,
    times: NDArray,
) -> BSpline:
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
        next_pulse_time = cumulative_time + pulse.duration
        pulse_times_idx = (times >= cumulative_time) & (times < next_pulse_time)
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
        cumulative_time = next_pulse_time

    # return pulse_waveforms
    return pulse_waveforms, make_interp_spline(
        times, pulse_waveforms, k=SPLINE_INTERP_ORDER
    )
