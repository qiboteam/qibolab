import numpy as np

from qibolab.pulse.pulse_shape.pulse_shape import PulseShape


class Gaussian(PulseShape):
    """
    Gaussian pulse shape.

    Args:
        pulse (Pulse): pulse associated with the shape
        rel_sigma (int): relative sigma so that the pulse standard deviation (sigma) = duration / rel_sigma
    
    .. math::

        A\exp^{-\\frac{1}{2}\\frac{(t-\mu)^2}{\sigma^2}}
    """

    def __init__(self, pulse, rel_sigma):
        self.name = "Gaussian"
        self.pulse = pulse
        self.rel_sigma = float(rel_sigma)

    @property
    def envelope_i(self):
        x = np.arange(0,self.pulse.duration,1)
        return self.pulse.amplitude * np.exp(-(1/2)*(((x-(self.pulse.duration-1)/2)**2)/(((self.pulse.duration)/self.rel_sigma)**2)))
        # same as: self.pulse.amplitude * gaussian(int(self.pulse.duration), std=int(self.pulse.duration/self.rel_sigma))

    @property
    def envelope_q(self):
        return np.zeros(int(self.pulse.duration))

    def __repr__(self):
        return f"{self.name}({self.rel_sigma})"