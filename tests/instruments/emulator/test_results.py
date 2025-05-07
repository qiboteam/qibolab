import numpy as np
import pytest

from qibolab._core.instruments.emulator.results import (
    calculate_probabilities_from_density_matrix,
)


def _order_probabilities(probs, qubits):
    """Arrange probabilities according to the given `qubits ordering."""
    return np.transpose(
        probs, [i for i, _ in sorted(enumerate(qubits), key=lambda t: t[1])]
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


def new_calculate(state, subsystems, nsubsystems, d):
    """Compute probabilities from density matrix."""
    state = np.reshape(state, 2 * nsubsystems * (d,))
    probs = np.abs(np.einsum(state, list(range(nsubsystems)) * 2, sorted(subsystems)))
    return _order_probabilities(probs, subsystems).ravel()


def former_apply_to_last_two_axes(func, array, *args, **kwargs):
    """Apply function over last two axes."""
    batch_shape = array.shape[:-2]
    m = array.shape[-1]
    reshaped_array = array.reshape(-1, m, m)
    processed = np.array([func(mat, *args, **kwargs) for mat in reshaped_array])
    return processed.reshape(*batch_shape, *processed.shape[1:])


def random_states(space: tuple[int, ...], sweeps: tuple[int, ...] = (), nacq: int = 1):
    dimension = np.prod(space, dtype=int)
    components = np.random.rand(*sweeps, nacq, dimension)

    state = components / np.sqrt((components**2).sum(axis=-1))[..., np.newaxis]

    return np.einsum("...i,...j->...ij", state, state)


def test_density_to_probs():
    density = random_states((3,) * 4)
    a = former_calculate(density, (1, 3), nsubsystems=4, d=3)
    b = new_calculate(density, (1, 3), nsubsystems=4, d=3)

    assert pytest.approx(a) == b


def test_apply_to_last_two_axes():
    densities = random_states((2,) * 4, (3, 2), nacq=2)
    a = former_apply_to_last_two_axes(
        new_calculate, densities, (1, 3), nsubsystems=4, d=2
    )
    b = calculate_probabilities_from_density_matrix(
        densities, (1, 3), nsubsystems=4, d=2
    )

    assert pytest.approx(a) == b
