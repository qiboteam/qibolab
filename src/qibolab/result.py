"""Common result operations."""

import numpy as np
import numpy.typing as npt

IQ = npt.NDArray[np.float64]
"""An array of I and Q values.

It is assumed that the I and Q component are discriminated by the
innermost dimension of the array.
"""

Result = npt.NDArray[np.float64]
"""An array of results."""


def _lift(values: IQ) -> npt.NDArray:
    """Transpose the innermost dimension to the outermost."""
    return np.moveaxis(values, -1, 0)


def _sink(values: npt.NDArray) -> IQ:
    """Transpose the outermost dimension to the innermost.

    Inverse of :func:`_lift`.
    """
    return np.moveaxis(values, 0, -1)


def collect(i: npt.NDArray, q: npt.NDArray) -> IQ:
    """Collect I and Q components in a single array."""
    return _sink(np.stack([i, q]))


def unpack(iq: IQ) -> tuple[npt.NDArray, npt.NDArray]:
    """Unpack I and Q components from single array.

    Inverse of :func:`collect`.
    """
    i, q = tuple(_lift(iq))
    return i, q


def magnitude(iq: IQ):
    """Signal magnitude.

    It is supposed to be a tension, possibly in arbitrary units.
    """
    iq_ = _lift(iq)
    return np.sqrt(iq_[0] ** 2 + iq_[1] ** 2)


def average(values: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Perform the values average.

    It returns both the average estimator itself, and its standard
    deviation estimator.

    Use this also for I and Q values in the *standard layout*, cf. :class:`IQ`.
    """
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    return mean, std


def average_iq(i: npt.NDArray, q: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Perform the average over I and Q.

    Convenience wrapper over :func:`average` for separate i and q samples arrays.
    """
    return average(collect(i, q))


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
    # The absolute value is only needed to make sure the result is always positive, even
    # when extremely close to zero
    return abs(1 - state - np.mean(values, axis=0))
