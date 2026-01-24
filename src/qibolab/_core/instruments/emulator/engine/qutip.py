from functools import cached_property
from typing import Union
from numpy.typing import NDArray

from .abstract import Operator, OperatorEvolution, SimulationEngine, EvolutionResult

__all__ = ["QutipEngine", "QutipCuquantumEngine"]


class QutipEngine(SimulationEngine):
    """Qutip simulation engine."""
    has_flipped_index: bool = False

    @cached_property
    def engine(self):
        """Return the qutip engine."""
        # TODO: maybe it can be improved
        import qutip as qt
        
        return qt

    def evolve(
        self,
        hamiltonian: Operator,
        initial_state: Operator,
        time: list[float],
        time_hamiltonian: OperatorEvolution = None,
        collapse_operators: list[Operator] = None,
        dimensions: dict = None,
        **kwargs,
    ):
        """Evolve the system."""
        if time_hamiltonian is not None:
            hamiltonian += self.engine.QobjEvo(time_hamiltonian.operators)
        return self.engine.mesolve(
            hamiltonian, initial_state, time, collapse_operators, **kwargs
        )

    def create(self, n: int, **kwargs) -> Operator:
        """Create operator for n levels system."""
        return self.engine.create(n)

    def destroy(self, n: int, **kwargs) -> Operator:
        """Destroy operator for n levels system."""
        return self.engine.destroy(n)

    def identity(self, n: int, **kwargs) -> Operator:
        """Identity operator for n levels system."""
        return self.engine.qeye(n)

    def tensor(self, operators: list[Operator]) -> Operator:
        """Tensor product of a list of operators."""
        return self.engine.tensor(*operators)

    def expand(self, op: Operator, targets: Union[int, list[int]], dims: list[int]):
        """Expand operator in larger Hilbert space."""
        return self.engine.expand_operator(op, targets, dims)

    def basis(self, dim: list, state: list) -> Operator:
        """Basis operator for n levels system."""
        return self.engine.basis(dimensions=dim, n=state)

    def get_state_dm(self, state: Operator, **kwargs) -> NDArray:
        if state.type=='ket':
            state = self.engine.ket2dm(state)
        return state.full()

    def get_evolution_states(self, results: EvolutionResult) -> list:
        return results.states

    def relaxation_op(self, transition: list, dim: int, **kwargs) -> Operator:
        return self.engine.basis(dim, transition[0]) * self.engine.basis(dim, transition[1]).dag()

    def dephasing_op(self, pair: list, dim: int, **kwargs) -> Operator:
        return self.engine.basis(dim, pair[0]) * self.engine.basis(dim, pair[0]).dag() - self.engine.basis(dim, pair[1]) * self.engine.basis(dim, pair[1]).dag()


class QutipCuquantumEngine(QutipEngine):
    """Qutip simulation engine using cuquantum."""
    
    @cached_property
    def engine(self):
        """Return the qutip engine using cuquantum."""
        import qutip as qt

        import qutip_cuquantum
        from cuquantum.densitymat import WorkStream
        stream = WorkStream()
        qutip_cuquantum.set_as_default(stream)
        
        return qt