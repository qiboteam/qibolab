from qutip import basis, destroy, tensor

N_LEVELS = 3
"""Levels for transmon system."""

STATE_0 = basis(N_LEVELS, 0)
"""State 0."""
STATE_1 = basis(N_LEVELS, 1)
"""State 1."""
STATE_2 = basis(N_LEVELS, 2)
"""State 2."""

INITIAL_STATE = tensor(STATE_0)
"""System initial state."""


def _probability(state: int):
    """Probability of qubit in state."""
    return basis(N_LEVELS, state) * basis(N_LEVELS, state).dag()


QUBIT_DESTROY = destroy(N_LEVELS)
"""Qubit destruction operator."""
QUBIT_CREATE = QUBIT_DESTROY.dag()
"""Qubit creation operator."""
QUBIT_NUMBER = QUBIT_CREATE * QUBIT_DESTROY
"""Qubit number operator."""
QUBIT_DRIVE = 1.0j * (QUBIT_CREATE - QUBIT_DESTROY)
"""Qubit drive term."""

# TODO: check these operators
L1 = STATE_0 * STATE_1.dag() + STATE_0 * STATE_2.dag()
"""Time relaxation operator."""
L2 = STATE_1 * STATE_1.dag() - STATE_2 * STATE_2.dag()
"""Dephasing operator."""
