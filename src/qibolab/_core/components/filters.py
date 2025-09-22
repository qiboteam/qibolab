from typing import List, Literal, Optional

import numpy as np

from ..serialize import Model

__all__ = ["Filters", "ExponentialFilter", "FiniteImpulseResponseFilter"]


class Filter(Model):
    """Filter class."""

    @property
    def feedback(self) -> Optional[List]:
        return None

    @property
    def feedforward(self) -> Optional[List]:
        return None


Filters = List[Filter]
"""List of filters."""


class ExponentialFilter(Filter):
    """Exponential filter."""

    kind: Literal["exp"] = "exp"
    amplitude: float
    tau: float
    sampling_rate: float = 1

    def _compute_filter(self):
        alpha = 1 - np.exp(-1 / (self.sampling_rate * self.tau * (1 + self.amplitude)))
        k = (
            self.amplitude / ((1 + self.amplitude) * (1 - alpha))
            if self.amplitude < 0
            else self.amplitude / (1 + self.amplitude - alpha)
        )
        b0 = 1 - k + k * alpha
        b1 = -(1 - k) * (1 - alpha)
        a0 = 1
        a1 = -(1 - alpha)
        return [a0, a1], [b0, b1]

    @property
    def feedback(self):
        return self._compute_filter()[0]

    @property
    def feedforward(self):
        return self._compute_filter()[1]


class FiniteImpulseResponseFilter(Filter):
    """FIR filter."""

    kind: Literal["fir"] = "fir"
    coefficients: List[float]

    @property
    def feedforward(self):
        return self.coefficients
