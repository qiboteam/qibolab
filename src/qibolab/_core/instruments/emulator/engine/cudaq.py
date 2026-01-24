from functools import cached_property, reduce
from typing import Union
import numpy as np
from numpy.typing import NDArray

from .abstract import Operator, OperatorEvolution, SimulationEngine, EvolutionResult

__all__ = ["CudaqEngine"]


class CudaqEngine(SimulationEngine):
    """CudaQ simulation engine."""
    max_qubit_dim: int = 2
    has_flipped_index: bool = True
    
    @cached_property
    def engine(self):
        """Return the cudaq engine set up with custom operators."""
        import cudaq
        cudaq.set_target("dynamics")

        def relax_op_mat(target_dim, transition):
            def op():
                op_mat = np.zeros([target_dim, target_dim], dtype=np.complex128)
                op_mat[transition[0]][transition[1]] = 1
                
                return op_mat
            return op
        
        def deph_op_mat(target_dim, pair):
            def op():
                op_mat = np.zeros([target_dim, target_dim], dtype=np.complex128)
                op_mat[pair[0]][pair[0]] = 1
                op_mat[pair[1]][pair[1]] = -1
                
                return op_mat
            return op
        
        for target_dim in range(2,self.max_qubit_dim+1):
            for transition in [[0,1]]:
                cudaq.operators.define(f"relax_op_{target_dim}_{transition[0]}_{transition[1]}", [target_dim], relax_op_mat(target_dim, transition))
            for pair in [[0,1]]:
                cudaq.operators.define(f"deph_op_{target_dim}_{pair[0]}_{pair[1]}", [target_dim], deph_op_mat(target_dim, pair))
            
        return cudaq
        
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
        # convert time to schedule
        schedule = self.engine.Schedule(time, ["t"])
        
        # TODO: check options for integrator

        """Evolve the system."""
        if time_hamiltonian is not None:        
            for op, waveform in time_hamiltonian.operators:
                hamiltonian += self.engine.ScalarOperator(waveform) * op
        
        return self.engine.evolve(
            hamiltonian,
            dimensions,
            schedule, 
            initial_state, 
            collapse_operators,
            store_intermediate_results=self.engine.IntermediateResultSave.ALL,
            integrator=self.engine.ScipyZvodeIntegrator(),
            **kwargs
        )

    def create(self, target: int, **kwargs) -> Operator:
        """Create operator of target Hilbert space index."""
        return self.engine.boson.create(target)

    def destroy(self, target: int, **kwargs) -> Operator:
        """Destroy operator of target Hilbert space index."""
        return self.engine.boson.annihilate(target)

    def identity(self, target: int, **kwargs) -> Operator:
        """Identity operator of target Hilbert space index."""
        return self.engine.boson.identity(target)

    def tensor(self, operators: list[Operator]) -> Operator:
        """Tensor product of a list of operators."""
        return reduce(lambda x, y: x * y, operators)

    def expand(self, op: Operator, targets: Union[int, list[int]], dims: list[int]):
        """Expand operator in larger Hilbert space - does nothing, feature built-in for cudaq, executed during runtime."""
        return op

    def basis(self, dim: int, state: int) -> Operator:
        """Basis state for n levels system."""
        statevec = np.zeros(np.prod(dim))
        state_index = get_index(dim, state)
        statevec[state_index] = 1.
        return self.engine.State.from_data(np.array(statevec, dtype=np.complex128))

    def get_state_dm(self, state: Operator, statevector_dimension:int = 0) -> NDArray:
        state_dm = self.engine.amplitudes(state)
        try:
            state_dm[0,1]
        except:
            state_dm_len = len(state_dm)
            if state_dm_len==statevector_dimension**2:
                state_dm = state_dm.reshape([statevector_dimension,statevector_dimension])
            else:
                state_dm = np.outer(state_dm,np.conjugate(state_dm))

        return state_dm

    def get_evolution_states(self, results: EvolutionResult) -> list:
        return results.intermediate_states()
    
    def relaxation_op(self, transition: list, target: int, dim: int, **kwargs) -> Operator:
        return self.engine.operators.instantiate(f"relax_op_{dim}_{transition[0]}_{transition[1]}", [target])

    def dephasing_op(self, pair: list, target: int, dim: int, **kwargs) -> Operator:
        return self.engine.operators.instantiate(f"deph_op_{dim}_{pair[0]}_{pair[1]}", [target])

def get_index(dim, state):
    dim = np.array(dim)

    # Compute all products dims[i+1:] using reverse cumulative product
    factors = np.cumprod(dim[::-1])[::-1][1:].tolist()
    factors.append(1)

    return int(np.dot(state, factors))