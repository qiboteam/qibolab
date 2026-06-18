"""Abstract engine for platform emulation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline, PPoly

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
    ) -> EvolutionResult:
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


def _spline_function(spline: BSpline):
    """Convert a SciPy spline into a JAX-traceable piecewise polynomial.

    SciPy ``BSpline.__call__`` cannot be traced by JAX, so the spline is
    converted once into its piecewise-polynomial representation and evaluated
    with a Horner scheme on the JAX side, preserving the cubic interpolation
    of the QuTiP engine exactly.
    """
    import jax.numpy as jnp

    polynomial = PPoly.from_spline(spline)
    breaks = jnp.asarray(polynomial.x)
    coefficients = jnp.asarray(polynomial.c)

    def evaluate(t):
        index = jnp.searchsorted(breaks, t, side="right") - 1
        index = jnp.clip(index, 0, breaks.size - 2)
        shifted_t = t - breaks[index]
        value = coefficients[0, index]
        for coefficient in coefficients[1:]:
            value = value * shifted_t + coefficient[index]
        return value

    return evaluate
