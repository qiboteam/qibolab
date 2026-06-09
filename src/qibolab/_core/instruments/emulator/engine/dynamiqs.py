from collections.abc import Iterable
from functools import cached_property
from pathlib import Path

import numpy as np

from .abstract import Operator, OperatorEvolution, SimulationEngine

__all__ = ["DynamiqsEngine"]


class DynamiqsEngine(SimulationEngine):
    """Dynamiqs simulation engine."""

    @cached_property
    def engine(self):
        """Return the dynamiqs module."""
        import dynamiqs as dq

        return dq

    def save_operators(self, operators, dump_dir: Path) -> None:
        """Persist the static operators once per experiment."""
        arrays = {f"op_{i}": np.asarray(op.to_jax()) for i, op in enumerate(operators)}
        np.savez(str(dump_dir / "operators.npz"), **arrays)

    def evolve(
        self,
        hamiltonian: Operator,
        initial_state: Operator,
        time: Iterable[float],
        time_hamiltonian: OperatorEvolution = None,
        collapse_operators: list[Operator] = None,
        **kwargs,
    ):
        """Evolve the system (open-system Lindblad master equation)."""
        H = hamiltonian

        if time_hamiltonian is not None:
            times = np.asarray(time_hamiltonian.times)
            # pwc needs (N+1) boundaries for N interval values; our samples and
            # grid are equal length, so append one trailing boundary (forward hold).
            dt = times[1] - times[0]
            boundaries = np.append(times, times[-1] + dt)
            for op, coeff in zip(
                time_hamiltonian.operators, time_hamiltonian.coefficients
            ):
                H = H + self.engine.pwc(boundaries, np.asarray(coeff), op)

        jump_ops = collapse_operators if collapse_operators is not None else []

        return self.engine.mesolve(
            H,
            jump_ops,
            initial_state,
            np.asarray(time),
            **kwargs,
        )

    def create(self, n: int) -> Operator:
        """Creation operator for n levels system."""
        return self.engine.create(n)

    def destroy(self, n: int) -> Operator:
        """Annihilation operator for n levels system."""
        return self.engine.destroy(n)

    def identity(self, n: int) -> Operator:
        """Identity operator for n levels system."""
        return self.engine.eye(n)

    def tensor(self, operators: list[Operator]) -> Operator:
        """Tensor product of a list of operators."""
        return self.engine.tensor(*operators)

    def expand(self, op: Operator, dims: list[int], target: int) -> Operator:
        """Expand an operator into a larger Hilbert space.

        Places `op` on subsystem `target`, identities on all others, and
        tensors them in `dims` order (matching dq.tensor's left-to-right
        convention and QuTiP's expand_operator).
        """
        factors = [
            op if i == target else self.engine.eye(dim) for i, dim in enumerate(dims)
        ]
        return self.engine.tensor(*factors)

    def basis(self, dim: int, state: int) -> Operator:
        """Basis state for n levels system."""
        return self.engine.basis(dim, state)
