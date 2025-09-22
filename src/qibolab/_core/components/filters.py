from typing import Annotated, List, Literal, Union

import numpy as np
from pydantic import Field

from ..serialize import Model

__all__ = ["Filters", "ExponentialFilter", "FiniteImpulseResponseFilter"]


class ExponentialFilter(Model):
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


class FiniteImpulseResponseFilter(Model):
    """FIR filter."""

    kind: Literal["fir"] = "fir"
    coefficients: List[float]

    @property
    def feedforward(self):
        return self.coefficients


Filter = Annotated[
    Union[ExponentialFilter, FiniteImpulseResponseFilter],
    Field(discriminator="kind"),
]
Filters = List[Filter]
"""List of filters."""
