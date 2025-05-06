"""Abstract engine for platform emulation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Protocol, Union

from ....serialize import Model

__all__ = ["SimulationEngine", "Operator", "TimeDependentOperator", "OperatorEvolution"]


class Operator(Protocol):
    """Abstract operator interface."""

    n: int
    """Dimension of Hilber space."""

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

    # TODO: Add method to return [op, time]


@dataclass
class OperatorEvolution:
    operators: list[Union[Operator, list[Operator, Callable[[float, dict], float]]]] = (
        field(default_factory=list)
    )

    def __add__(
        self, other: Union[Operator, "OperatorEvolution"]
    ) -> "OperatorEvolution":
        """Add two operator evolutions."""
        cls = self.__class__
        if isinstance(other, list):
            return cls(other.operators + self.operators)
        return cls([other] + self.operators)


class SimulationEngine(Model, ABC):
    """Parent class for generic simulation engine."""

    @property
    @abstractmethod
    def engine(self):
        """The name of the type of fruit."""

    @abstractmethod
    def evolve(
        self,
        hamiltonian: Operator,
        initial_state: Operator,
        time: list[float],
        collapse_operators: list[Operator] = None,
    ):
        """Evolve the system."""
        raise NotImplementedError

    @abstractmethod
    def create(self, n: int) -> Operator:
        """Create operator for n levels system."""
        raise NotImplementedError

    @abstractmethod
    def destroy(self, n: int) -> Operator:
        """Destroy operator for n levels system."""
        raise NotImplementedError

    @abstractmethod
    def identity(self, n: int) -> Operator:
        """Identity operator for n levels system."""
        raise NotImplementedError

    @abstractmethod
    def tensor(self, operators: list[Operator]) -> Operator:
        """Tensor product of a list of operators."""
        raise NotImplementedError

    @abstractmethod
    def expand(self, op: Operator, targets: Union[int, list[int]], dims: list[int]):
        """Expand operator in larger Hilbert space."""
        raise NotImplementedError

    @abstractmethod
    def basis(self, n: int, state: int) -> Operator:
        """Basis operator for n levels system."""
        raise NotImplementedError
