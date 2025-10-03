from typing import Annotated, Literal, Union

import numpy as np
from pydantic import Field

from ..serialize import Model

__all__ = ["ExponentialFilter", "FiniteImpulseResponseFilter"]


class ExponentialFilter(Model):
    """Exponential filter.

    Correct filters for signal with behavior
    1 + amplitude * e^(-t/tau).
    """

    kind: Literal["exp"] = "exp"
    amplitude: float
    """Amplitude for exponential term."""
    tau: float
    """Time decay in exponential in samples unit."""

    def _compute_filter(self) -> tuple[list[float], list[float]]:
        """Computing feedback and feedforward taps."""
        alpha = 1 - np.exp(-1 / (self.tau * (1 + self.amplitude)))
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
    def feedback(self) -> list[float]:
        return self._compute_filter()[0]

    @property
    def feedforward(self) -> list[float]:
        return self._compute_filter()[1]


class FiniteImpulseResponseFilter(Model):
    """FIR filter."""

    kind: Literal["fir"] = "fir"
    coefficients: list[float]

    @property
    def feedforward(self):
        return self.coefficients


Filter = Annotated[
    Union[ExponentialFilter, FiniteImpulseResponseFilter],
    Field(discriminator="kind"),
]
