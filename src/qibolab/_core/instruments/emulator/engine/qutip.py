from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from .abstract import Operator, OperatorEvolution, SimulationEngine

__all__ = ["QutipEngine"]

INTEGRATION_MAX_TIME_STEP = 0.02
"""ns, min resolution of the integrator"""
INTEGRATION_MULTIPLIER = 200
"""factor for computing max number of steps for the ode solver"""
INTEGRATION_MIN_TIME_STEP = 5e-3
"""ns, max resolution of the integrator"""

HAMILTONIAN_FILENAME = "System_Hamiltonian"
STATE_FILENAME = "State_Evolution"


class QutipEngine(SimulationEngine):
    """Qutip simulation engine."""

    @cached_property
    def engine(self):
        """Return the qutip engine."""
        # TODO: maybe it can be improved
        import qutip as qt

        return qt

    def dump_results(
        self, hamiltonian: Operator, sim_results: Any, dump_dir: Path
    ) -> None:
        """Save the Hamiltonian and simulation results to files with incremented naming."""

        dump_dir.mkdir(parents=True, exist_ok=True)

        count_1 = sum(
            1
            for file in dump_dir.iterdir()
            if file.is_file() and HAMILTONIAN_FILENAME in file.name
        )
        count_2 = sum(
            1
            for file in dump_dir.iterdir()
            if file.is_file() and STATE_FILENAME in file.name
        )
        count = max(count_1, count_2)

        self.engine.qsave(
            hamiltonian, str(dump_dir) + f"/{HAMILTONIAN_FILENAME}_{count}"
        )
        self.engine.qsave(sim_results, str(dump_dir) + f"/{STATE_FILENAME}_{count}")

    def evolve(
        self,
        hamiltonian: Operator,
        initial_state: Operator,
        time: Iterable[float],
        time_hamiltonian: OperatorEvolution = None,
        collapse_operators: list[Operator] = None,
        save_evolution: Path | None = None,
        **kwargs,
    ):
        """Evolve the system."""

        time_diff = np.diff(time)
        nsteps = max(time_diff) / INTEGRATION_MIN_TIME_STEP * INTEGRATION_MULTIPLIER
        # not every SciPy solvers accepts as parameters min_step, that's why we
        # define nsteps instead
        options = {"max_step": INTEGRATION_MAX_TIME_STEP, "nsteps": nsteps}

        if time_hamiltonian is not None:
            hamiltonian = [hamiltonian] + [list(o) for o in time_hamiltonian.operators]

        sim_results = self.engine.mesolve(
            hamiltonian,
            initial_state,
            time,
            collapse_operators,
            options=options,
            **kwargs,
        )

        if save_evolution is not None:
            self.dump_results(
                hamiltonian=hamiltonian,
                sim_results=sim_results,
                dump_dir=save_evolution,
            )

        return sim_results

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

    def expand(self, op: Operator, dims: list[int], targets: int | list[int]):
        """Expand operator in larger Hilbert space."""
        return self.engine.expand_operator(op, dims, targets)

    def basis(self, dim: int, state: int) -> Operator:
        """Basis operator for n levels system."""
        return self.engine.basis(dimensions=dim, n=state)
