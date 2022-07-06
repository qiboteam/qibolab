"""Rectangular pulse shape."""
from dataclasses import dataclass

import numpy as np

from qibolab.pulse.pulse_shape.pulse_shape import PulseShape
from qibolab.typings import PulseShapeName
from qibolab.utils import Factory


@Factory.register
@dataclass
class Rectangular(PulseShape):
    """Rectangular/square pulse shape."""

    name = PulseShapeName.RECTANGULAR

    def envelope(self, duration: int, amplitude: float, resolution: float = 1.0):
        """Constant amplitude envelope.

        Args:
            duration (int): Duration of the pulse (ns).
            amplitude (float): Maximum amplitude of the pulse.

        Returns:
            ndarray: Amplitude of the envelope for each time step.
        """
        return amplitude * np.ones(round(duration / resolution))

    def __repr__(self):
        return f"{self.name}()"