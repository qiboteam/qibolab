import json
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path

import numpy as np
from scipy.interpolate import make_interp_spline

from .abstract import (
    HAMILTONIAN_FILENAME,
    SIMULATOR_CONFIG,
    SWEEP_SIMULATION_FILENAME,
    Operator,
    OperatorEvolution,
    SimulationEngine,
)

__all__ = ["QutipEngine"]

SPLINE_INTERP_ORDER = 3
"""Polynomial order used for interpolating the pulses with a spline function."""
INTEGRATION_MAX_TIME_STEP = 0.02
"""ns, min resolution of the integrator"""
INTEGRATION_MULTIPLIER = 200
"""factor for computing max number of steps for the ode solver"""
INTEGRATION_MIN_TIME_STEP = 5e-3
"""ns, max resolution of the integrator"""


class QutipEngine(SimulationEngine):
    """Qutip simulation engine."""

    @cached_property
    def engine(self):
        """Return the qutip engine."""
        # TODO: maybe it can be improved
        import qutip as qt

        return qt

    def evolve(
        self,
        hamiltonian: Operator,
        initial_state: Operator,
        time: Iterable[float],
        time_hamiltonian: OperatorEvolution = None,
        collapse_operators: list[Operator] = None,
        **kwargs,
    ):
        """Evolve the system."""

        time_diff = np.diff(time)
        nsteps = max(time_diff) / INTEGRATION_MIN_TIME_STEP * INTEGRATION_MULTIPLIER
        # not every SciPy solvers accepts as parameters min_step, that's why we
        # define nsteps instead
        options = {"max_step": INTEGRATION_MAX_TIME_STEP, "nsteps": nsteps}

        if time_hamiltonian is not None:
            evo_times = time_hamiltonian.times
            hamiltonian = [hamiltonian] + [
                [op, make_interp_spline(x=evo_times, y=c, k=SPLINE_INTERP_ORDER)]
                for op, c in time_hamiltonian.operators
            ]

        sim_results = self.engine.mesolve(
            hamiltonian,
            initial_state,
            time,
            collapse_operators,
            options=options,
            **kwargs,
        )

        return sim_results, options

    def create(self, n: int) -> Operator:
        """Create operator for n levels system."""
        return self.engine.create(n)

    def destroy(self, n: int) -> Operator:
        """Destroy operator for n levels system."""
        return self.engine.destroy(n)

    def identity(self, n: int) -> Operator:
        """Identity operator for n levels system."""
        return self.engine.qeye(n)

    def tensor(self, operators: list[Operator]) -> Operator:
        """Tensor product of a list of operators."""
        return self.engine.tensor(*operators)

    def expand(self, op: Operator, targets: int | list[int], dims: list[int]):
        """Expand operator in larger Hilbert space."""
        return self.engine.expand_operator(op, targets, dims)

    def basis(self, dim: int, state: int) -> Operator:
        """Basis operator for n levels system."""
        return self.engine.basis(dimensions=dim, n=state)


def load_simulation(
    simulation_path: Path | str, sequence_index: int, sweep_index: int
) -> tuple[OperatorEvolution, np.typing.NDArray, np.typing.NDArray, dict]:
    """Load a saved Qutip simulation from disk given a specific pulse sequence and a specific index of the parameter sweep to load
    (used to select the correct time-coefficients and state files).
    """

    if isinstance(simulation_path, str):
        simulation_path = Path(simulation_path)

    simulated_sequence_path = simulation_path / f"sequence_{sequence_index}"
    if not simulated_sequence_path.is_dir():
        raise NotADirectoryError()

    hamiltonians = np.load(simulated_sequence_path / (HAMILTONIAN_FILENAME + ".npy"))
    sweep_sim = np.load(
        simulated_sequence_path / (SWEEP_SIMULATION_FILENAME + f"_{sweep_index}.npz")
    )
    result_states = sweep_sim["results"]
    time_coeffs = sweep_sim["time_coeffs"]

    with open(simulated_sequence_path / (SIMULATOR_CONFIG + ".json")) as f:
        sim_configs = json.load(f)

    system = [hamiltonians[0]] + [
        [ham, coeffs] for ham, coeffs in zip(hamiltonians[1:], time_coeffs[1:])
    ]
    timesteps = time_coeffs[0]

    return system, result_states, timesteps, sim_configs
