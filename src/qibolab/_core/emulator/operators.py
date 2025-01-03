from qutip import destroy, sigmaz

GHZ_TO_HZ = 1e9
"""Converting GHz to Hz."""
HZ_TO_GHZ = 1e-9
"""Converting Hz to GHz."""

QUBIT_DESTROY = destroy(2)
"""Qubit destruction operator."""
QUBIT_CREATE = QUBIT_DESTROY.dag()
"""Qubit creation operator."""
QUBIT_NUMBER = QUBIT_CREATE * QUBIT_DESTROY
"""Qubit number operator."""
QUBIT_DRIVE = QUBIT_CREATE + QUBIT_DESTROY
"""Qubit drive term."""
SIGMAZ = sigmaz()
"""Sigma z operator."""
