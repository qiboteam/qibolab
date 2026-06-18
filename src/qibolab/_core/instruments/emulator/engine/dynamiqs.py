"""Dynamiqs simulation engine."""

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.interpolate import make_interp_spline

from .abstract import (
    INTEGRATION_MIN_TIME_STEP,
    INTEGRATION_MULTIPLIER,
    Operator,
    OperatorEvolution,
    SimulationEngine,
    jax_interpolation,
)

__all__ = ["DynamiqsEngine"]

SPLINE_INTERP_ORDER = 3


@dataclass
class DynamiqsOperator:
    """Compatibility wrapper for Dynamiqs operators.

    The emulator combines operators with QuTiP semantics, where ``*`` is the
    matrix product. Dynamiqs reserves ``*`` for scalar/element-wise
    multiplication and uses ``@`` for the matrix product, so the wrapper
    dispatches ``*`` between two operators to ``@`` on the underlying qarrays.
    """

    qarray: Any

    def dag(self) -> "DynamiqsOperator":
        return type(self)(self.qarray.dag())

    def full(self) -> np.ndarray:
        return np.asarray(self.qarray.to_jax())

    def __add__(self, other):
        if isinstance(other, int) and other == 0:
            return self
        return type(self)(self.qarray + _unwrap(other))

    def __radd__(self, other):
        if isinstance(other, int) and other == 0:
            return self
        return type(self)(_unwrap(other) + self.qarray)

    def __sub__(self, other):
        return type(self)(self.qarray - _unwrap(other))

    def __rsub__(self, other):
        return type(self)(_unwrap(other) - self.qarray)

    def __mul__(self, other):
        other = _unwrap(other)
        if _is_qarray(other):
            return type(self)(self.qarray @ other)
        return type(self)(self.qarray * other)

    def __rmul__(self, other):
        other = _unwrap(other)
        if _is_qarray(other):
            return type(self)(other @ self.qarray)
        return type(self)(other * self.qarray)

    def __matmul__(self, other):
        return type(self)(self.qarray @ _unwrap(other))

    def __rmatmul__(self, other):
        return type(self)(_unwrap(other) @ self.qarray)

    def __truediv__(self, other):
        return type(self)(self.qarray / other)

    def __neg__(self):
        return type(self)(-self.qarray)


@dataclass
class DynamiqsEvolutionResult:
    """Evolution result exposing states compatible with Qibolab."""

    result: Any

    @property
    def states(self) -> list[DynamiqsOperator]:
        return [DynamiqsOperator(state) for state in self.result.states]


class DynamiqsEngine(SimulationEngine):
    """Dynamiqs simulation engine."""

    precision: Literal["single", "double"] = "double"
    """Floating point precision used by Dynamiqs/JAX."""
    device: Literal["cpu", "gpu", "tpu"] = "cpu"
    """Device used by Dynamiqs/JAX arrays."""
    device_index: int = 0
    """Index of the selected JAX device."""
    matmul_precision: Literal["low", "high", "highest"] = "highest"
    """Matrix multiplication precision used by JAX on accelerators."""
    method: Literal["adaptive", "fixed"] = "adaptive"
    """Integration method.

    ``adaptive`` uses the Tsit5 solver with `rtol`/`atol` step control, where
    the sub-resolution steps are inferred by the solver itself. ``fixed`` uses
    the Rouchon3 master-equation solver with a constant step `fixed_step_dt`,
    providing a guaranteed time resolution analogous to the QuTiP engine's
    ``max_step`` option.
    """
    rtol: float = 1e-8
    """Relative tolerance of the adaptive integrator."""
    atol: float = 1e-8
    """Absolute tolerance of the adaptive integrator."""
    fixed_step_dt: float = INTEGRATION_MIN_TIME_STEP
    """ns, integration step of the fixed-step integrator.

    Defaults to the finest resolution targeted by the QuTiP engine, which is
    enough to resolve the lab-frame qubit oscillations of the bundled
    platforms with the third-order Rouchon solver.
    """

    @cached_property
    def engine(self):
        """Return the dynamiqs engine."""
        import dynamiqs as dq

        dq.set_precision(self.precision)
        dq.set_device(self.device, self.device_index)
        dq.set_matmul_precision(self.matmul_precision)
        return dq

    def _method(self, time: np.ndarray):
        """Build the integration method honoring the engine configuration."""
        if self.method == "fixed":
            return self.engine.method.Rouchon3(dt=self.fixed_step_dt)
        # mirror the QuTiP engine `nsteps` bound: total duration over the
        # finest resolution, scaled by the same safety multiplier
        duration = max(time[-1] - time[0], INTEGRATION_MIN_TIME_STEP)
        max_steps = int(duration / INTEGRATION_MIN_TIME_STEP * INTEGRATION_MULTIPLIER)
        return self.engine.method.Tsit5(
            rtol=self.rtol, atol=self.atol, max_steps=max(max_steps, 100_000)
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

        hamiltonian = self.engine.constant(_unwrap(hamiltonian))
        if time_hamiltonian is not None:
            times = time_hamiltonian.times
            coefficients = [
                make_interp_spline(times, coefficient, k=SPLINE_INTERP_ORDER)
                for coefficient in time_hamiltonian.coefficients
            ]
            for operator, coefficient in zip(
                time_hamiltonian.operators, coefficients, strict=True
            ):
                hamiltonian += self.engine.modulated(
                    jax_interpolation(coefficient), _unwrap(operator)
                )

        method = kwargs.pop("method", self._method(time))
        options = kwargs.pop("options", self.engine.Options(progress_meter=False))
        result = self.engine.mesolve(
            hamiltonian,
            [_unwrap(op) for op in collapse_operators or []],
            _unwrap(initial_state),
            time,
            method=method,
            options=options,
            **kwargs,
        )

        if save_evolution is not None:
            self.dump_results(
                hamiltonian=hamiltonian,
                sim_results=result,
                dump_dir=save_evolution,
            )

        return DynamiqsEvolutionResult(result)

    def create(self, n: int) -> Operator:
        """Create operator for n levels system."""
        return DynamiqsOperator(self.engine.create(n))

    def destroy(self, n: int) -> Operator:
        """Destroy operator for n levels system."""
        return DynamiqsOperator(self.engine.destroy(n))

    def identity(self, n: int) -> Operator:
        """Identity operator for n levels system."""
        return DynamiqsOperator(self.engine.eye(n))

    def tensor(self, operators: list[Operator]) -> Operator:
        """Tensor product of a list of operators."""
        return DynamiqsOperator(self.engine.tensor(*[_unwrap(op) for op in operators]))

    def expand(
        self, op: Operator, targets: int | list[int], dims: list[int]
    ) -> Operator:
        """Expand operator in larger Hilbert space."""
        # The current emulator calls this method with QuTiP's argument order:
        # ``expand_operator(op, dims, targets)``.
        dimensions = list(targets)
        target_subsystems = [dims] if isinstance(dims, int) else list(dims)
        op_array = np.asarray(_unwrap(op).to_jax())
        return DynamiqsOperator(
            self.engine.asqarray(
                _expand_operator(op_array, targets=target_subsystems, dims=dimensions),
                dims=tuple(dimensions),
            )
        )

    def basis(self, dim: int | list[int], state: int | list[int]) -> Operator:
        """Basis operator for n levels system."""
        return DynamiqsOperator(self.engine.basis(dim, state))

    def save_operators(self) -> None:
        """Persist the static operators once per experiment."""
        return


def _unwrap(operator):
    return operator.qarray if isinstance(operator, DynamiqsOperator) else operator


def _is_qarray(operator) -> bool:
    return hasattr(operator, "to_jax") and hasattr(operator, "dag")


def _expand_operator(op: np.ndarray, targets: list[int], dims: list[int]) -> np.ndarray:
    """Dense equivalent of qutip.expand_operator for small emulator systems."""
    nsubsystems = len(dims)
    target_dims = [dims[target] for target in targets]
    expected_shape = 2 * (int(np.prod(target_dims)),)
    if op.shape != expected_shape:
        raise ValueError(
            f"Operator shape {op.shape} is incompatible with {target_dims}."
        )

    tensor = op.reshape(target_dims + target_dims)
    identities = [np.eye(dim, dtype=op.dtype) for dim in dims]
    identity_targets = [i for i in range(nsubsystems) if i not in targets]

    expanded = tensor
    input_axes = dict(zip(targets, range(len(targets)), strict=True))
    output_axes = dict(zip(targets, range(len(targets), 2 * len(targets)), strict=True))
    for target in identity_targets:
        expanded = np.tensordot(expanded, identities[target], axes=0)
        input_axes[target] = expanded.ndim - 2
        output_axes[target] = expanded.ndim - 1

    permutation = [input_axes[i] for i in range(nsubsystems)] + [
        output_axes[i] for i in range(nsubsystems)
    ]
    return np.transpose(expanded, permutation).reshape(2 * (int(np.prod(dims)),))
