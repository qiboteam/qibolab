from qutip import basis, sigmap, sigmax, sigmaz, tensor

N_LEVELS = 2
"""Levels for transmon system."""

STATE_0 = basis(N_LEVELS, 0)
"""State 0."""
STATE_1 = basis(N_LEVELS, 1)
"""State 1."""

INITIAL_STATE = tensor(STATE_0)
"""System initial state."""

SIGMAZ = sigmaz()
"""Qubit destruction operator."""
QUBIT_DRIVE = sigmax()
"""Qubit drive term."""

# TODO: check these operators
L1 = sigmap()
"""Time relaxation operator."""
L2 = sigmaz()
"""Dephasing operator."""
