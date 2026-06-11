from collections.abc import Iterable
from functools import cached_property
from pathlib import Path

import numpy as np
import jax.numpy as jnp


from .abstract import Operator, OperatorEvolution, SimulationEngine

__all__ = ["DynamiqsEngine"]


class DynamiqsEngine(SimulationEngine):
    """Dynamiqs simulation engine."""

    @cached_property
    def engine(self):
        """Return the dynamiqs module."""
        import dynamiqs as dq

        return dq

    def save_operators(self, operators, dump_dir: Path) -> None:
        """Persist the static operators once per experiment."""
        arrays = {f"op_{i}": np.asarray(op.to_jax()) for i, op in enumerate(operators)}
        np.savez(str(dump_dir / "operators.npz"), **arrays)

    def evolve(
        self,
        hamiltonian: Operator,
        initial_state: Operator,
        time: Iterable[float],
        time_hamiltonian: OperatorEvolution = None,
        collapse_operators: list[Operator] = None,
        **kwargs,
    ):
        """Evolve the system (open-system Lindblad master equation)."""
        H = hamiltonian

        if time_hamiltonian is not None:
            times = np.asarray(time_hamiltonian.times)
            # pwc needs (N+1) boundaries for N interval values; our samples and
            # grid are equal length, so append one trailing boundary (forward hold).
            dt = times[1] - times[0]
            boundaries = np.append(times, times[-1] + dt)
            for td in time_hamiltonian.operators:
                H = H + self.engine.pwc(boundaries, np.asarray(td.coefficients), td.operator)

        jump_ops = collapse_operators if collapse_operators is not None else []

        return self.engine.mesolve(
            H,
            jump_ops,
            initial_state,
            np.asarray(time),
            **kwargs,
        )

    def create(self, n: int) -> Operator:
        """Creation operator for n levels system."""
        return self.engine.create(n)

    def destroy(self, n: int) -> Operator:
        """Annihilation operator for n levels system."""
        return self.engine.destroy(n)

    def identity(self, n: int) -> Operator:
        """Identity operator for n levels system."""
        return self.engine.eye(n)

    def tensor(self, operators: list[Operator]) -> Operator:
        """Tensor product of a list of operators."""
        return self.engine.tensor(*operators)

    def expand(self, op, dims, targets):
        """Expand an operator onto subsystem(s) `targets` in the full `dims` space.

        Faithful port of qutip.expand_operator + Qobj.permute (gather convention).
        Handles single-subsystem (targets: int) and multi-subsystem (targets: list).
        Reads op's own subsystem structure from op.dims rather than assuming it.
        """
        dq = self.engine
        N = len(dims)
        if isinstance(targets, int):
            targets = [targets]

        # --- non-target slots ---
        rest_pos = [q for q in range(N) if q not in targets]
        id_list = [dq.eye(dims[i]) for i in rest_pos]

        # --- tensor(op, *identities): op's subsystems occupy the FRONT slots ---
        tensored = dq.tensor(op, *id_list) if id_list else op

        # --- read op's OWN subsystem structure (ground truth, not assumed) ---
        # op.dims should be the per-subsystem tuple, e.g. (2, 2) for a 2-transmon op.
        op_dims = list(op.dims)
        # sanity: op should span exactly len(targets) subsystems
        assert len(op_dims) == len(targets), (
            f"op spans {len(op_dims)} subsystems but {len(targets)} targets given"
        )

        rest_dims = [dims[i] for i in rest_pos]
        # layout in `tensored`: [op subsystem 0, ..., op subsystem k-1, rest...]
        current_dims = op_dims + rest_dims

        # --- new_order, exactly as qutip.expand_operator ---
        new_order = [0] * N
        for i, t in enumerate(targets):
            new_order[t] = i
        rest_qubits = list(range(len(targets), N))
        for i, ind in enumerate(rest_pos):
            new_order[ind] = rest_qubits[i]

        # --- apply permute via reshape/transpose (gather convention) ---
        mat = jnp.asarray(tensored.to_jax())
        # (D, D) -> per-subsystem row axes + per-subsystem col axes, in CURRENT order
        mat = mat.reshape(current_dims + current_dims)
        # output axis i <- input axis new_order[i] (gather; matches qutip .permute and jnp.transpose)
        row_perm = list(new_order)
        col_perm = [N + k for k in new_order]
        mat = jnp.transpose(mat, row_perm + col_perm)
        # back to (D, D) in natural subsystem order
        D = int(np.prod(dims))
        mat = mat.reshape(D, D)

        return dq.asqarray(mat)

    def basis(self, dim: int, state: int) -> Operator:
        """Basis state for n levels system."""
        return self.engine.basis(dim, state)