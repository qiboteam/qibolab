from functools import cached_property
from typing import Literal, Union

from .abstract import Operator, OperatorEvolution, SimulationEngine

__all__ = ["QutipEngine"]


class QutipEngine(SimulationEngine):
    kind: Literal["qutip"] = "qutip"

    @cached_property
    def engine(self):
        """Return the qutip engine."""
        import qutip as qt

        return qt

    def evolve(
        self,
        hamiltonian: Operator,
        initial_state: Operator,
        time: list[float],
        time_hamiltonian: OperatorEvolution = None,
        collapse_operators: list[Operator] = None,
    ):
        """Evolve the system."""
        if time_hamiltonian is not None:
            hamiltonian += self.engine.QobjEvo(time_hamiltonian.operators)
        return self.engine.mesolve(hamiltonian, initial_state, time, collapse_operators)

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

    def expand(self, op: Operator, targets: Union[int, list[int]], dims: list[int]):
        """Expand operator in larger Hilbert space."""
        return self.engine.expand_operator(op, targets, dims)

    def basis(self, dim: int, state: int) -> Operator:
        """Basis operator for n levels system."""
        return self.engine.basis(dimensions=dim, n=state)
