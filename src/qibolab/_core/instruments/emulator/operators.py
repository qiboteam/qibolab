"""Useful operators in qutip for n levels transmon."""

from qutip import (
    Qobj,
    QobjEvo,
    basis,
    create,
    destroy,
    expand_operator,
    mesolve,
    qeye,
    tensor,
)

Operator = Qobj
"""Generic operator."""
TimeDependentOperator = QobjEvo
"""Generic time dependent operator."""
evolve = mesolve
"""Generic time evolution function."""


def transmon_create(n: int) -> Qobj:
    """Creation operator for n levels system."""
    return create(n)


def transmon_destroy(n: int) -> Qobj:
    """Destruction operator for n levels system."""
    return destroy(n)


identity = qeye
"""Identity operator for n levels system."""

tensor_product = tensor
"""Tensor product of a list of operators using Qutip."""

expand = expand_operator
"""Expand operator in larger Hilber space using Qutip."""


def relaxation(final_state: int, initial_state: int, n: int) -> Qobj:
    """Relaxation operator.

    Matrix element for initial_state -> final_state decay.
    """
    return basis(n, final_state) * basis(n, initial_state).dag()


def dephasing(state0: int, state1: int, n: int) -> Qobj:
    """Dephasing operator between two states."""
    return (
        basis(n, state0) * basis(n, state0).dag()
        - basis(n, state1) * basis(n, state1).dag()
    )


def probability(state: int, n: int) -> Qobj:
    """Probability of measuring state."""
    return basis(n, state) * basis(n, state).dag()


def state(state, n) -> Qobj:
    """State as tensor for qutip."""
    return tensor(basis(n, state))
