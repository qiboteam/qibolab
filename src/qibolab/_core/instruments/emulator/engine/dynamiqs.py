"""Dynamiqs engine for platform emulation."""

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import BSpline

from .abstract import Operator, OperatorEvolution, SimulationEngine
from .qutip import (
    HAMILTONIAN_FILENAME,
    INTEGRATION_MIN_TIME_STEP,
    INTEGRATION_MULTIPLIER,
    STATE_FILENAME,
)

__all__ = ["DynamiqsEngine"]


@dataclass(frozen=True)
class DynamiqsOperator:
    """QuTiP-compatible wrapper over a Dynamiqs qarray."""

    raw: Any

    @property
    def n(self) -> int:
        """Dimension of the Hilbert space."""
        return max(self.raw.shape[-2:])

    @property
    def dims(self):
        """Hilbert space dimensions."""
        return self.raw.dims

    @property
    def shape(self):
        """Operator shape."""
        return self.raw.shape

    def dag(self) -> "DynamiqsOperator":
        """Return the adjoint of the operator."""
        return type(self)(self.raw.dag())

    def full(self) -> np.ndarray:
        """Return a dense NumPy representation."""
        import dynamiqs as dq

        return dq.to_numpy(self.raw)

    def __array__(self, dtype=None):
        array = self.full()
        return array.astype(dtype) if dtype is not None else array

    @staticmethod
    def _unwrap(other: Any) -> Any:
        return other.raw if isinstance(other, DynamiqsOperator) else other

    @staticmethod
    def _is_zero(other: Any) -> bool:
        return isinstance(other, (int, float, complex)) and other == 0

    def __add__(self, other: Any) -> "DynamiqsOperator":
        if self._is_zero(other):
            return self
        return type(self)(self.raw + self._unwrap(other))

    def __radd__(self, other: Any) -> "DynamiqsOperator":
        if self._is_zero(other):
            return self
        return type(self)(self._unwrap(other) + self.raw)

    def __sub__(self, other: Any) -> "DynamiqsOperator":
        return type(self)(self.raw - self._unwrap(other))

    def __rsub__(self, other: Any) -> "DynamiqsOperator":
        return type(self)(self._unwrap(other) - self.raw)

    def __neg__(self) -> "DynamiqsOperator":
        return type(self)(-self.raw)

    def __mul__(self, other: Any) -> "DynamiqsOperator":
        if isinstance(other, DynamiqsOperator):
            return type(self)(self.raw @ other.raw)
        return type(self)(self.raw * other)

    def __rmul__(self, other: Any) -> "DynamiqsOperator":
        if isinstance(other, DynamiqsOperator):
            return type(self)(other.raw @ self.raw)
        return type(self)(other * self.raw)

    def __matmul__(self, other: Any) -> "DynamiqsOperator":
        return type(self)(self.raw @ self._unwrap(other))

    def __rmatmul__(self, other: Any) -> "DynamiqsOperator":
        return type(self)(self._unwrap(other) @ self.raw)

    def __truediv__(self, other: Any) -> "DynamiqsOperator":
        return type(self)(self.raw / other)


@dataclass(frozen=True)
class DynamiqsEvolutionResult:
    """QuTiP-compatible result wrapper for Dynamiqs evolutions."""

    raw: Any
    states: list[DynamiqsOperator]


def unwrap(operator: Any) -> Any:
    """Return the native Dynamiqs object for wrapped operators."""
    return operator.raw if isinstance(operator, DynamiqsOperator) else operator


def _bspline_prefactor(spline: BSpline):
    """Convert a SciPy B-spline into a JAX-compatible scalar function."""
    import jax.numpy as jnp

    knots = jnp.asarray(spline.t)
    coefficients = jnp.asarray(spline.c)
    degree = spline.k

    def safe_divide(numerator, denominator):
        return jnp.where(denominator == 0, 0, numerator / denominator)

    def prefactor(time):
        basis = ((knots[:-1] <= time) & (time < knots[1:])).astype(coefficients.dtype)

        for order in range(1, degree + 1):
            size = knots.size - 1 - order
            left = safe_divide(
                time - knots[:size],
                knots[order : order + size] - knots[:size],
            )
            right = safe_divide(
                knots[order + 1 : order + 1 + size] - time,
                knots[order + 1 : order + 1 + size] - knots[1 : 1 + size],
            )
            basis = left * basis[:size] + right * basis[1 : size + 1]

        value = basis @ coefficients[: basis.size]
        return jnp.where(time == knots[-1], coefficients[-1], value)

    return prefactor


class DynamiqsEngine(SimulationEngine):
    """Dynamiqs simulation engine."""

    @cached_property
    def engine(self):
        """Return the Dynamiqs engine."""
        import dynamiqs as dq

        dq.set_precision("double")
        return dq

    def _wrap(self, operator: Any) -> DynamiqsOperator:
        return DynamiqsOperator(operator)

    def _time_hamiltonian(
        self, hamiltonian: Operator, time_hamiltonian: OperatorEvolution | None
    ):
        hamiltonian = self.engine.constant(unwrap(hamiltonian))
        if time_hamiltonian is None:
            return hamiltonian

        for operator, time in time_hamiltonian.operators:
            hamiltonian += self.engine.modulated(
                _bspline_prefactor(time),
                unwrap(operator),
            )

        return hamiltonian

    def dump_results(
        self,
        hamiltonian: Any,
        sim_results: Any,
        time: np.ndarray,
        dump_dir: Path,
    ) -> None:
        """Save the Hamiltonian and simulation results to NumPy files."""
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

        hamiltonian_values = np.stack(
            [self.engine.to_numpy(hamiltonian(t)) for t in time]
        )
        np.savez(
            dump_dir / f"{HAMILTONIAN_FILENAME}_{count}.npz",
            time=time,
            hamiltonian=hamiltonian_values,
        )
        np.savez(
            dump_dir / f"{STATE_FILENAME}_{count}.npz",
            time=np.asarray(sim_results.tsave),
            states=self.engine.to_numpy(sim_results.states),
        )

    def evolve(
        self,
        hamiltonian: Operator,
        initial_state: Operator,
        time: Iterable[float],
        time_hamiltonian: OperatorEvolution = None,
        collapse_operators: list[Operator] = None,
        save_evolution: Path | None = None,
        **kwargs,
    ) -> DynamiqsEvolutionResult:
        """Evolve the system."""
        time = np.asarray(list(time), dtype=float)
        time_diff = np.diff(time)
        nsteps = int(
            max(time_diff, default=INTEGRATION_MIN_TIME_STEP)
            / INTEGRATION_MIN_TIME_STEP
            * INTEGRATION_MULTIPLIER
        )
        method = kwargs.pop(
            "method",
            self.engine.method.Tsit5(
                rtol=1e-8,
                atol=1e-8,
                max_steps=max(nsteps, 100000),
            ),
        )
        options = kwargs.pop(
            "options",
            self.engine.Options(progress_meter=False),
        )

        hamiltonian = self._time_hamiltonian(hamiltonian, time_hamiltonian)
        collapse_operators = [unwrap(op) for op in (collapse_operators or [])]

        sim_results = self.engine.mesolve(
            hamiltonian,
            collapse_operators,
            self.engine.todm(unwrap(initial_state)),
            time,
            method=method,
            options=options,
            **kwargs,
        )

        if save_evolution is not None:
            self.dump_results(
                hamiltonian=hamiltonian,
                sim_results=sim_results,
                time=time,
                dump_dir=save_evolution,
            )

        states = [self._wrap(sim_results.states[i]) for i in range(len(time))]
        return DynamiqsEvolutionResult(raw=sim_results, states=states)

    def create(self, n: int) -> DynamiqsOperator:
        """Create operator for n levels system."""
        return self._wrap(self.engine.create(n))

    def destroy(self, n: int) -> DynamiqsOperator:
        """Destroy operator for n levels system."""
        return self._wrap(self.engine.destroy(n))

    def identity(self, n: int) -> DynamiqsOperator:
        """Identity operator for n levels system."""
        return self._wrap(self.engine.eye(n))

    def tensor(self, operators: list[Operator]) -> DynamiqsOperator:
        """Tensor product of a list of operators."""
        return self._wrap(self.engine.tensor(*(unwrap(op) for op in operators)))

    def expand(
        self, op: Operator, dims: list[int], targets: int | list[int]
    ) -> DynamiqsOperator:
        """Expand operator in larger Hilbert space."""
        dims = list(dims)
        targets = [targets] if isinstance(targets, int) else list(targets)
        target_dims = [dims[target] for target in targets]
        untouched = [i for i in range(len(dims)) if i not in targets]
        untouched_dims = [dims[target] for target in untouched]

        operator = self.engine.to_numpy(unwrap(op)).reshape(
            np.prod(target_dims, dtype=int),
            np.prod(target_dims, dtype=int),
        )

        if untouched_dims:
            compact = np.kron(operator, np.eye(np.prod(untouched_dims, dtype=int)))
        else:
            compact = operator

        order = targets + untouched
        position = {target: i for i, target in enumerate(order)}
        shape = [dims[i] for i in order] + [dims[i] for i in order]
        permutation = [position[i] for i in range(len(dims))] + [
            len(dims) + position[i] for i in range(len(dims))
        ]
        expanded = (
            compact.reshape(shape)
            .transpose(permutation)
            .reshape(
                np.prod(dims, dtype=int),
                np.prod(dims, dtype=int),
            )
        )

        return self._wrap(self.engine.asqarray(expanded, dims=tuple(dims)))

    def basis(self, dim: int | list[int], state: int | list[int]) -> DynamiqsOperator:
        """Basis operator for n levels system."""
        if isinstance(dim, list):
            dim = tuple(dim)
        return self._wrap(self.engine.basis(dim, state))
