from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import Any

from .abstract import EvolutionResult, Operator, OperatorEvolution, SimulationEngine

__all__ = ["DynamiqsEngine"]

INTEGRATION_MAX_TIME_STEP = 0.02
"""ns, min resolution of the integrator"""
INTEGRATION_MULTIPLIER = 200
"""factor for computing max number of steps for the ode solver"""
INTEGRATION_MIN_TIME_STEP = 5e-3
"""ns, max resolution of the integrator"""

HAMILTONIAN_FILENAME = "System_Hamiltonian"
STATE_FILENAME = "State_Evolution"


class CompResult(EvolutionResult):
    def __init__(self, states):
        self.states = states


class DynamiqsEngine(SimulationEngine):
    """Dynamiqs simulation engine."""

    @cached_property
    def engine(self):
        """Return the dynamiqs engine."""
        # TODO: maybe it can be improved
        import dynamiqs as dq

        return dq

    def dump_results(
        self, hamiltonian: Operator, sim_results: Any, dump_dir: Path
    ) -> None:
        """Save the Hamiltonian and simulation results to files with incremented naming."""

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

        import qutip

        qutip.qsave(hamiltonian, str(dump_dir) + f"/{HAMILTONIAN_FILENAME}_{count}")
        qutip.qsave(sim_results, str(dump_dir) + f"/{STATE_FILENAME}_{count}")

    def load_results(
        self, dump_dir: Path, count=None
    ) -> tuple[tuple[Operator, OperatorEvolution], Any]:
        """Load the Hamiltonian and simulation results from file."""
        # if count is not given, load the latest results (with highest count)
        if not isinstance(dump_dir, Path):
            dump_dir = Path(dump_dir)
        if count is None:
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
            count = max(count_1, count_2) - 1

        import qutip

        [hamiltonian, time_hamiltonian] = qutip.qload(
            str(dump_dir) + f"/{HAMILTONIAN_FILENAME}_{count}"
        )
        sim_results = qutip.qload(str(dump_dir) + f"/{STATE_FILENAME}_{count}")
        return [hamiltonian, time_hamiltonian], sim_results

    def evolve(
        self,
        hamiltonian: Operator,
        initial_state: Operator,
        time: Iterable[float],
        time_hamiltonian: OperatorEvolution = None,
        collapse_operators: list[Operator] = None,
        save_evolution: Path | None = None,
        **kwargs,
    ):
        """Evolve the system."""

        # Using an integrator with a fixed time step
        method = self.engine.method.Rouchon3(dt=INTEGRATION_MIN_TIME_STEP)

        H = hamiltonian

        if time_hamiltonian is not None:
            import jax

            # Linear jax interpolation (splines not currently supported in JAX)
            H = self.engine.timecallable(
                lambda t: (
                    self.engine.asqarray(hamiltonian)
                    + sum(
                        [
                            self.engine.asqarray(op[0]).elmul(
                                jax.numpy.interp(t, op[1], op[2])
                            )
                            for op in time_hamiltonian.operators
                        ]
                    )
                )
            )

        sim_results = self.engine.mesolve(
            H=H,
            jump_ops=collapse_operators,
            rho0=initial_state,
            tsave=time,
            method=method,
            **kwargs,
        )
        comp_results = CompResult(sim_results.states.to_qutip())

        if save_evolution is not None:
            if isinstance(H, self.engine.QArray):
                hamiltonian = hamiltonian.to_qutip()
            self.dump_results(
                hamiltonian=[hamiltonian, time_hamiltonian],
                sim_results=comp_results,
                dump_dir=save_evolution,
            )

        return comp_results

    # Converting all Dynamiqs objects to Qutip for compatibility:
    # Qutip uses the * symbol to multiply operators, while in Dynamiqs,
    # the @ symbol is used for matrix multiplication, and the * symbol
    # is reserved for element-wise multiplication with a scalar.
    def create(self, n: int) -> Operator:
        """Create operator for n levels system."""
        return self.engine.create(n).to_qutip()

    def destroy(self, n: int) -> Operator:
        """Destroy operator for n levels system."""
        return self.engine.destroy(n).to_qutip()

    def identity(self, n: int) -> Operator:
        """Identity operator for n levels system."""
        return self.engine.eye(n).to_qutip()

    def tensor(self, operators: list[Operator]) -> Operator:
        """Tensor product of a list of operators."""
        return self.engine.tensor(*operators).to_qutip()

    def expand(self, op: Operator, targets: int | list[int], dims: list[int]):
        """Expand operator in larger Hilbert space."""
        # parameters in hamiltonian.py:
        # op, self.dims, self.hilbert_space_index(i)
        if isinstance(targets, int) or len(targets) == 1:
            return op
        # print(op, targets, dims)
        actual_N = len(targets)
        actual_targets = dims
        actual_dims = targets
        if not isinstance(actual_targets, list):
            actual_targets = [actual_targets]

        # from qutip source code:
        new_order = [0] * actual_N
        for i, t in enumerate(actual_targets):
            new_order[t] = i
        # allocate the rest qutbits (not targets) to the empty
        # position in new_order
        rest_pos = [q for q in list(range(actual_N)) if q not in actual_targets]
        rest_qubits = list(range(len(actual_targets), actual_N))
        for i, ind in enumerate(rest_pos):
            new_order[ind] = rest_qubits[i]
        id_list = [self.identity(actual_dims[i]) for i in rest_pos]
        return self.tensor([op] + id_list).permute(new_order)

    def basis(self, dim: int, state: int) -> Operator:
        """Basis operator for n levels system."""
        return self.engine.basis(dim=dim, number=state).to_qutip()
