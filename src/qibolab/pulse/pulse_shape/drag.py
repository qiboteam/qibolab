"""Drag pulse shape."""
from dataclasses import dataclass

import numpy as np

from qibolab.pulse.pulse_shape.pulse_shape import PulseShape
from qibolab.typings import PulseShapeName
from qibolab.utils import Factory


@Factory.register
@dataclass
class Drag(PulseShape):
    """Derivative Removal by Adiabatic Gate (DRAG) pulse shape."""

    name = PulseShapeName.DRAG

    num_sigmas: float
    beta: float

    def envelope(self, duration: int, amplitude: float, resolution: float = 1.0):
        """DRAG envelope centered with respect to the pulse.

        Args:
            duration (int): Duration of the pulse (ns).
            amplitude (float): Maximum amplitude of the pulse.

        Returns:
            ndarray: Amplitude of the envelope for each time step.
        """
        sigma = duration / self.num_sigmas
        time = np.arange(duration / resolution) * resolution
        mu_ = duration / 2
        gaussian = amplitude * np.exp(-0.5 * (time - mu_) ** 2 / sigma**2)
        gaussian = (gaussian - gaussian[0]) / (1 - gaussian[0])  # Shift to avoid introducing noise at time 0
        return gaussian + 1j * self.beta * (-(time - mu_) / sigma**2) * gaussian

    def __repr__(self):
        return f"{self.name}({self.rel_sigma}, {self.beta})"