import numpy as np
import numpy.typing as npt


def _transpose(values: npt.NDArray):
    """Transpose the innermost dimension to the outermost."""
    return np.transpose(values, [-1, *range(values.ndim - 1)])


def _backspose(values: npt.NDArray):
    """Innermost transposition inverse.

    Cf. :func:`_transpose`.
    """
    return np.transpose(values, [*range(1, values.ndim), 0])


def magnitude(iq: npt.NDArray):
    """Signal magnitude.

    It is supposed to be a tension, possibly in arbitrary units.

    It is assumed that the I and Q component are discriminated by the
    innermost dimension of the array.
    """
    iq_ = _transpose(iq)
    return np.sqrt(iq_[0] ** 2 + iq_[1] ** 2)


def phase(iq: npt.NDArray):
    """Signal phase in radians.

    It is assumed that the I and Q component are discriminated by the
    innermost dimension of the array.
    """
    iq_ = _transpose(iq)
    return np.unwrap(np.arctan2(iq_[0], iq_[1]))


def probability(values: npt.NDArray, state: int = 0):
    """Returns the statistical frequency of the specified state.

    The only accepted values `state` are `0` and `1`.
    """
    return abs(1 - state - np.mean(values, axis=0))
