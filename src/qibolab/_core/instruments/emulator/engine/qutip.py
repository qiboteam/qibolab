from collections.abc import Iterable
from functools import cached_property
from pathlib import Path

import numpy as np
from scipy.interpolate import make_interp_spline

from .abstract import Operator, OperatorEvolution, SimulationEngine
from .evolution_dump import (
    EvolutionDump,
    SPLINE_ORDER,
    dump_evolution,
    load_evolution_dump,
)

__all__ = ["EvolutionDump", "QutipEngine"]

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

    def save_operators(self, operators, dump_dir: Path) -> None:
        """Persist Hamiltonian operators once per experiment."""
        dump_dir.mkdir(parents=True, exist_ok=True)
        self.engine.qsave(operators, str(dump_dir / "operators"))

    def dump_evolution(
        self,
        dump_dir: Path,
        *,
        operators,
        time_grid: np.ndarray,
        time_coefficients: np.ndarray | None,
        density_matrices: np.ndarray,
    ) -> Path:
        """Save operators, coefficients, and density matrices in a structured dump."""
        return dump_evolution(
            dump_dir,
            engine=self.engine,
            operators=operators,
            time_grid=time_grid,
            time_coefficients=time_coefficients,
            density_matrices=density_matrices,
        )

    def load_evolution_dump(self, dump_dir: Path) -> EvolutionDump:
        """Load a structured evolution dump."""
        return load_evolution_dump(dump_dir, engine=self.engine)

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
            spline_order = min(SPLINE_ORDER, len(time_hamiltonian.times) - 1)
            pairs = [
                [
                    operator,
                    make_interp_spline(
                        time_hamiltonian.times, coefficient, k=spline_order
                    ),
                ]
                for operator, coefficient in zip(
                    time_hamiltonian.operators, time_hamiltonian.coefficients
                )
            ]
            hamiltonian = [hamiltonian] + pairs

        return self.engine.mesolve(
            hamiltonian,
            initial_state,
            time,
            collapse_operators,
            options=options,
            **kwargs,
        )

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
