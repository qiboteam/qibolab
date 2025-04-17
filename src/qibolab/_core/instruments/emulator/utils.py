import numpy as np
from numpy.typing import NDArray


def ndchoice(probabilities: NDArray, samples: int) -> NDArray:
    """Sample elements with n-dimensional probabilities.

    This is the n-dimensional version of :func:`np.random.choice`, which instead of
    vectorizing over the picked elements, it assumes them to be just a plain integer
    range (which in turn could be used to index a suitable array, if relevant), while it
    allows the probabilities to be higher dimensional.

    The ``probabilities`` argument specifies the set of probabilities, which are
    intended to be normalized arrays over the innermost dimension. Such that the whole
    array describe a set of ``probabilities.shape[:-1]`` discrete distributions.

    `samples` is instead the number of samples to extract from each distribution.

    .. seealso::

        Generalized from https://stackoverflow.com/a/47722393, which presents the
        two-dimensional version.
    """
    return (
        probabilities.cumsum(-1).reshape(*probabilities.shape, -1)
        > np.random.rand(*probabilities.shape[:-1], 1, samples)
    ).argmax(-2)


def shots(probabilities: NDArray, nshots: int) -> NDArray:
    """Extract shots from state |0> ... |n> probabilities.

    This function just wraps :func:`ndchoice`, taking care of creating the n-D array of
    binomial distributions, and extracting shots as the outermost dimension.
    """
    shots = ndchoice(probabilities, nshots)
    # move shots from innermost to outermost dimension
    return np.moveaxis(shots, -1, 0)


def _order_probabilities(probs, qubits, nqubits):
    """Arrange probabilities according to the given `qubits ordering."""
    unmeasured, reduced = [], {}
    for i in range(nqubits):
        if i in qubits:
            reduced[i] = i - len(unmeasured)
        else:
            unmeasured.append(i)
    return np.transpose(probs, [reduced.get(i) for i in qubits])


def calculate_probabilities_density_matrix(state, subsystems, nsubsystems, d):
    """Compute probabilities from density matrix."""
    order = tuple(sorted(subsystems))
    order += tuple(i for i in range(nsubsystems) if i not in subsystems)
    order = order + tuple(i + nsubsystems for i in order)

    shape = 2 * (d ** len(subsystems), d ** (nsubsystems - len(subsystems)))

    state = np.reshape(state, 2 * nsubsystems * (d,))
    state = np.reshape(np.transpose(state, order), shape)

    probs = np.abs(np.einsum("abab->a", state))
    probs = np.reshape(probs, len(subsystems) * (d,))

    return _order_probabilities(probs, subsystems, nsubsystems).ravel()


def apply_to_last_two_axes(func, array, *args, **kwargs):
    """Apply function over last two axes."""
    batch_shape = array.shape[:-2]
    m = array.shape[-1]
    reshaped_array = array.reshape(-1, m, m)
    processed = np.array([func(mat, *args, **kwargs) for mat in reshaped_array])
    return processed.reshape(*batch_shape, *processed.shape[1:])
