"""Abstract engine for platform emulation."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from qibolab._core.serialize import Model

__all__ = [
    "SimulationEngine",
    "Operator",
    "TimeDependentOperator",
    "OperatorEvolution",
]

HAMILTONIAN_FILENAME = "System_Hamiltonian"
SWEEP_SIMULATION_FILENAME = "Time_Coefficients_and_Results"
SIMULATOR_CONFIG = "Simulator_Configs"

SPLINE_INTERP_ORDER = 3
"""Polynomial order used for interpolating the pulses with a spline function."""

INTEGRATION_MAX_TIME_STEP = 0.02
"""ns, min resolution of the integrator"""
INTEGRATION_MULTIPLIER = 200
"""factor for computing max number of steps for the ode solver"""
INTEGRATION_MIN_TIME_STEP = 5e-3
"""ns, max resolution of the integrator"""


class Operator(Protocol):
    """Abstract operator interface.

    It can represents both operators and states in the Hilbert space.
    """

    n: int
    """Dimension of Hilbert space."""

    def dag(self) -> "Operator":
        """Return the adjoint of the operator."""

    def __add__(self, other: "Operator") -> "Operator":
        """Add two operators."""


TimeDependentOperator = tuple[Operator, NDArray]
"""Abstract time dependent operator type."""


class EvolutionResult(Protocol):
    """Result from evolution."""

    states: list[Operator]
    """List of Operators."""


@dataclass
class OperatorEvolution:
    """Abstract operator evolution interface."""

    operators: list[Operator | TimeDependentOperator] = field(default_factory=list)
    """List of static or time-dependent operators for evolution."""
    times: NDArray = field(default_factory=lambda: np.array([], dtype=float))
    """Evolution times with time step equal to the waveforms resolution."""


class SimulationEngine(Model, ABC):
    """Parent class for generic simulation engine."""

    @property
    @abstractmethod
    def engine(self):
        """Engine module."""

    @abstractmethod
    def evolve(
        self,
        hamiltonian: Operator,
        initial_state: Operator,
        time: list[float],
        collapse_operators: list[Operator] = None,
        **kwargs,
    ) -> tuple[EvolutionResult, dict]:
        """Evolve the system."""

    @abstractmethod
    def create(self, n: int) -> Operator:
        """Create operator for n levels system."""

    @abstractmethod
    def destroy(self, n: int) -> Operator:
        """Destroy operator for n levels system."""

    @abstractmethod
    def identity(self, n: int) -> Operator:
        """Identity operator for n levels system."""

    @abstractmethod
    def tensor(self, operators: list[Operator]) -> Operator:
        """Tensor product of a list of operators."""

    @abstractmethod
    def expand(
        self, op: Operator, targets: int | list[int], dims: list[int]
    ) -> Operator:
        """Expand operator in larger Hilbert space."""

    @abstractmethod
    def basis(self, n: int, state: int) -> Operator:
        """Basis operator for n levels system."""


def jax_interpolation(
    spline_x: NDArray, spline_y: NDArray
) -> Callable[[NDArray], Iterable[float]]:
    """Convert a SciPy spline into a JAX-traceable piecewise polynomial.

    SciPy ``BSpline.__call__`` cannot be traced by JAX, so the points and coefficients are
    evaluated with a Horner scheme on the JAX side, preserving the cubic interpolation
    of the QuTiP engine exactly.
    """

    from diffrax import CubicInterpolation, backward_hermite_coefficients

    spline_c = backward_hermite_coefficients(spline_x, spline_y)

    return CubicInterpolation(spline_x, spline_c).evaluate
