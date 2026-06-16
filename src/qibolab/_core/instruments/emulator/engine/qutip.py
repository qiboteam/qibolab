from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from .abstract import Operator, OperatorEvolution, SimulationEngine

__all__ = ["EvolutionDump", "QutipEngine"]

INTEGRATION_MAX_TIME_STEP = 0.02
"""ns, min resolution of the integrator"""
INTEGRATION_MULTIPLIER = 200
"""factor for computing max number of steps for the ode solver"""
INTEGRATION_MIN_TIME_STEP = 5e-3
"""ns, max resolution of the integrator"""

HAMILTONIAN_FILENAME = "System_Hamiltonian"
STATE_FILENAME = "State_Evolution"


@dataclass(frozen=True)
class EvolutionDump:
    """Saved Hamiltonian and state evolution data."""

    path: Path
    """Directory containing the dump files."""
    hamiltonian: Any
    """Saved time-dependent Hamiltonian."""
    states: Any
    """Saved state evolution results."""


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
    ) -> Path:
        """Save the Hamiltonian and simulation results to a structured folder."""

        run_dir = self._create_dump_run_dir(dump_dir)

        self.engine.qsave(hamiltonian, str(run_dir / HAMILTONIAN_FILENAME))
        self.engine.qsave(sim_results, str(run_dir / STATE_FILENAME))

        return run_dir

    def load_results(self, dump_dir: Path) -> EvolutionDump:
        """Load a saved Hamiltonian and state evolution dump."""

        run_dir = self._resolve_dump_run_dir(dump_dir)

        hamiltonian = self.engine.qload(
            self._qload_path(run_dir / f"{HAMILTONIAN_FILENAME}.qu")
        )
        states = self.engine.qload(self._qload_path(run_dir / f"{STATE_FILENAME}.qu"))

        return EvolutionDump(
            path=run_dir,
            hamiltonian=hamiltonian,
            states=states,
        )

    @staticmethod
    def _create_dump_run_dir(dump_dir: Path) -> Path:
        """Create a unique run folder without counting existing files."""

        dump_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        for collision_count in range(10):
            suffix = "" if collision_count == 0 else f"-{collision_count:04d}"
            run_dir = dump_dir / f"run-{timestamp}{suffix}"
            try:
                run_dir.mkdir()
            except FileExistsError:
                continue
            return run_dir
        raise RuntimeError(f"Could not create a unique dump directory in {dump_dir}.")

    @staticmethod
    def _resolve_dump_run_dir(dump_dir: Path) -> Path:
        """Resolve a dump root or a concrete run directory to a run directory."""

        if _has_qutip_dump_files(dump_dir):
            return dump_dir

        run_dirs = sorted(
            path
            for path in dump_dir.iterdir()
            if path.is_dir() and _has_qutip_dump_files(path)
        )
        if len(run_dirs) == 0:
            raise FileNotFoundError(f"No qutip evolution dumps found in {dump_dir}.")
        return run_dirs[-1]

    @staticmethod
    def _qload_path(path: Path) -> str:
        """Return the filename format expected by qutip.qload."""

        if path.suffix == ".qu":
            return str(path.with_suffix(""))
        return str(path)

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
            hamiltonian = [hamiltonian] + time_hamiltonian.operators

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

    def expand(self, op: Operator, targets: int | list[int], dims: list[int]):
        """Expand operator in larger Hilbert space."""
        return self.engine.expand_operator(op, targets, dims)

    def basis(self, dim: int, state: int) -> Operator:
        """Basis operator for n levels system."""
        return self.engine.basis(dimensions=dim, n=state)


def _has_qutip_dump_files(path: Path) -> bool:
    """Return whether a directory has the fixed qutip dump artifacts."""

    return (path / f"{HAMILTONIAN_FILENAME}.qu").is_file() and (
        path / f"{STATE_FILENAME}.qu"
    ).is_file()
