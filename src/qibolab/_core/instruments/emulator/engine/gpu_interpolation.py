"""GPU interpolation utilities for efficient spline evaluation in JAX.

This module provides tools to convert SciPy splines into JAX-traceable
piecewise polynomial functions, enabling GPU-accelerated interpolation
while preserving numerical accuracy of cubic spline interpolation.
"""

from collections.abc import Callable, Iterable

from numpy.typing import NDArray
from scipy.interpolate import BSpline, PPoly


def jax_interpolation(spline: BSpline) -> Callable[[NDArray], Iterable[float]]:
    """Convert a SciPy spline into a JAX-traceable piecewise polynomial.

    SciPy ``BSpline.__call__`` cannot be traced by JAX, so the points and coefficients are
    evaluated with a Horner scheme on the JAX side, preserving the cubic interpolation
    of the QuTiP engine exactly.
    """
    # TODO: check if there is a more efficient implementation
    import jax.numpy as jnp

    polynomial = PPoly.from_spline(spline)
    breaks = jnp.asarray(polynomial.x)
    coefficients = jnp.asarray(polynomial.c)

    def evaluate(t):
        index = jnp.searchsorted(breaks, t, side="right") - 1
        index = jnp.clip(index, 0, breaks.size - 2)
        shifted_t = t - breaks[index]
        value = coefficients[0, index]
        for coefficient in coefficients[1:]:
            value = value * shifted_t + coefficient[index]
        return value

    return evaluate
