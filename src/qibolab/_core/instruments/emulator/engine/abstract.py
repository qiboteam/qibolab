"""Abstract engine for platform emulation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

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

    def full(self) -> "Operator":
        """Return the matrix form of the operator."""

    def __add__(self, other: "Operator") -> "Operator":
        """Add two operators."""

    def __sub__(self, other: "Operator") -> "Operator":
        """Subtract two operators."""

    def __mul__(self, other: float | int | complex) -> "Operator":
        """Scalar multiplication."""

    def __rmul__(self, other: float | int | complex) -> "Operator":
        """Right-hand scalar multiplication."""

    def __truediv__(self, other: float | int | complex) -> "Operator":
        """Scalar division."""

    def __matmul__(self, other: "Operator") -> "Operator":
        """Multiply two operators."""


TimeDependentOperator = tuple[Operator, NDArray]
"""Abstract time dependent operator type."""


class EvolutionResult(Protocol):
    """Result from evolution."""

    states: list[Operator]
    """List of Operators."""


@dataclass
class OperatorEvolution:
    """Abstract operator evolution interface."""

    static: Operator
    """static term in the system evolution"""
    operators: list[TimeDependentOperator] = field(default_factory=list)
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
        hamiltonian: OperatorEvolution,
        initial_state: Operator,
        time: ArrayLike,
        collapse_operators: list[Operator] | None = None,
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
        self, op: Operator, dims: list[int], targets: int | list[int]
    ) -> Operator:
        """Expand operator in larger Hilbert space."""

    @abstractmethod
    def basis(self, dim: int | list[int], state: int | list[int]) -> Operator:
        """Basis operator for n levels system."""
