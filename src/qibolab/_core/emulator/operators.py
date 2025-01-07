from qutip import basis, destroy

STATE_0 = basis(3, 0)
"""State 0."""
STATE_1 = basis(3, 1)
"""State 1."""
STATE_2 = basis(3, 2)
"""State 2."""

QUBIT_DESTROY = destroy(3)
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
