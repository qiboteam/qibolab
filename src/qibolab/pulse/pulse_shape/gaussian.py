from dataclasses import dataclass
import numpy as np

from qibolab.pulse.pulse_shape.pulse_shape import PulseShape
from qibolab.typings.enums import PulseShapeName
from qibolab.utils.factory import Factory


@Factory.register
@dataclass
class Gaussian(PulseShape):
    """
    Gaussian pulse shape.

    Args:
        pulse (Pulse): pulse associated with the shape
        num_sigmas (int): relative sigma so that the pulse standard deviation (sigma) = duration / num_sigmas
    
    .. math::

        A\exp^{-\\frac{1}{2}\\frac{(t-\mu)^2}{\sigma^2}}
    """

    name = PulseShapeName.GAUSSIAN
    num_sigmas: float

    def envelope(self, duration: int, amplitude: float, resolution: float = 1.0):
        """Gaussian envelope centered with respect to the pulse.

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
        return (gaussian - gaussian[0]) / (1 - gaussian[0])  # Shift to avoid introducing noise at time 0

    def __repr__(self):
        return f"{self.name}({self.num_sigmas})"