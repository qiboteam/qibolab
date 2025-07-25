"""Abstract engine for platform emulation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Protocol, Union

from ....serialize import Model

__all__ = ["SimulationEngine", "Operator", "TimeDependentOperator", "OperatorEvolution"]


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


class TimeDependentOperator(Protocol):
    """Abstract time dependent operator interface."""

    operator: Operator
    """Operator."""
    time: Callable[[float, dict], float]
    """Time function."""


class EvolutionResult(Protocol):
    """Result from evolution."""

    states: list[Operator]
    """List of Operators."""


@dataclass
class OperatorEvolution:
    """Abstract operator evolution interface."""

    operators: list[Union[Operator, TimeDependentOperator]] = field(
        default_factory=list
    )


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
        self, op: Operator, targets: Union[int, list[int]], dims: list[int]
    ) -> Operator:
        """Expand operator in larger Hilbert space."""

    @abstractmethod
    def basis(self, n: int, state: int) -> Operator:
        """Basis operator for n levels system."""
