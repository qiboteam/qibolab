"""Common result operations."""

import numpy as np
import numpy.typing as npt

IQ = npt.NDArray[np.float64]
"""An array of I and Q values.

It is assumed that the I and Q component are discriminated by the
innermost dimension of the array.
"""


def _lift(values: IQ) -> npt.NDArray:
    """Transpose the innermost dimension to the outermost."""
    return np.transpose(values, [-1, *range(values.ndim - 1)])


def _lower(values: npt.NDArray) -> IQ:
    """Transpose the outermost dimension to the innermost.

    Inverse of :func:`_transpose`.
    """
    return np.transpose(values, [*range(1, values.ndim), 0])


def magnitude(iq: IQ):
    """Signal magnitude.

    It is supposed to be a tension, possibly in arbitrary units.
    """
    iq_ = _lift(iq)
    return np.sqrt(iq_[0] ** 2 + iq_[1] ** 2)


def average(iq: IQ) -> tuple[npt.NDArray, npt.NDArray]:
    """Perform the average over i and q.

    It returns both the average estimator itself, and its standard
    deviation estimator.
    """
    mean = np.mean(iq, axis=0)
    std = np.std(iq, axis=0, ddof=1) / np.sqrt(iq.shape[0])
    return mean, std


def average_iq(i: npt.NDArray, q: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Perform the average over i and q.

    Wraps :func:`average` for separate i and q samples arrays.
    """
    return average(_lower(np.stack([i, q])))


def phase(iq: npt.NDArray):
    """Signal phase in radians.

    It is assumed that the I and Q component are discriminated by the
    innermost dimension of the array.
    """
    iq_ = _lift(iq)
    return np.unwrap(np.arctan2(iq_[0], iq_[1]))


def probability(values: npt.NDArray, state: int = 0):
    """Return the statistical frequency of the specified state.

    The only accepted values `state` are `0` and `1`.
    """
    return abs(1 - state - np.mean(values, axis=0))
