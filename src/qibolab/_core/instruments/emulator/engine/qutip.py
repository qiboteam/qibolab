from functools import cached_property
from typing import Iterable, Union

import numpy as np

from .abstract import Operator, OperatorEvolution, SimulationEngine

INTEGRATION_MAX_TIME_STEP = 0.02  # ns, min resolution of the integrator
INTEGRATION_MULTIPLIER = (
    100  # factor for computing max number of steps for the ode solver
)
INTEGRATION_MIN_TIME_STEP = 5e-3  # ns, max resolution of the integrator

__all__ = ["QutipEngine"]


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
        save_evolution: bool = False,
        **kwargs,
    ):
        """Evolve the system."""

        time_diff = np.diff(time)
        max_step = min(min(time_diff), INTEGRATION_MAX_TIME_STEP)
        min_step = (
            INTEGRATION_MIN_TIME_STEP
            if max_step > INTEGRATION_MIN_TIME_STEP
            else max_step / 10
        )
        nsteps = max(time_diff) / min_step * INTEGRATION_MULTIPLIER
        options = {"max_step": max_step, "nsteps": nsteps}

        if time_hamiltonian is not None:
            hamiltonian = [hamiltonian] + time_hamiltonian.operators

        sim_results = self.engine.mesolve(
            hamiltonian,
            initial_state,
            time,
            collapse_operators,
            options=options,
            **kwargs,
        )

        if save_evolution:
            from pathlib import Path

            directory = Path(".")
            filename_1 = "System_Hamiltonian"
            filename_2 = "State_Evolution"
            count_1 = sum(
                1
                for file in directory.iterdir()
                if file.is_file() and filename_1 in file.name
            )
            count_2 = sum(
                1
                for file in directory.iterdir()
                if file.is_file() and filename_2 in file.name
            )
            count = max(count_1, count_2)

            self.engine.qsave(hamiltonian, f"{filename_1}_{count}")
            self.engine.qsave(sim_results, f"{filename_2}_{count}")

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

    def expand(self, op: Operator, targets: Union[int, list[int]], dims: list[int]):
        """Expand operator in larger Hilbert space."""
        return self.engine.expand_operator(op, targets, dims)

    def basis(self, dim: int, state: int) -> Operator:
        """Basis operator for n levels system."""
        return self.engine.basis(dimensions=dim, n=state)
