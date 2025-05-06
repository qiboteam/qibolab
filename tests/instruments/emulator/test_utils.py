import numpy as np
import pytest

from qibolab._core.instruments.emulator.utils import (
    _order_probabilities,
    calculate_probabilities_density_matrix,
)


def former_calculate(state, subsystems, nsubsystems, d):
    """Compute probabilities from density matrix."""
    order = tuple(sorted(subsystems))
    order += tuple(i for i in range(nsubsystems) if i not in subsystems)
    order = order + tuple(i + nsubsystems for i in order)

    shape = 2 * (d ** len(subsystems), d ** (nsubsystems - len(subsystems)))

    state = np.reshape(state, 2 * nsubsystems * (d,))
    state = np.reshape(np.transpose(state, order), shape)

    probs = np.abs(np.einsum("abab->a", state))
    probs = np.reshape(probs, len(subsystems) * (d,))

    return _order_probabilities(probs, subsystems).ravel()


def random_states(space: tuple[int, ...], sweeps: tuple[int, ...] = (), nacq: int = 1):
    dimension = np.prod(space, dtype=int)
    components = np.random.rand(*sweeps, nacq, dimension)

    state = components / np.sqrt((components**2).sum(axis=-1))

    return np.einsum("...i,...j->...ij", state, state)


def test_density_to_probs():
    density = random_states((2,) * 4)
    a = former_calculate(density, (1, 3), nsubsystems=4, d=2)
    b = calculate_probabilities_density_matrix(density, (1, 3), nsubsystems=4, d=2)

    assert pytest.approx(a) == b
