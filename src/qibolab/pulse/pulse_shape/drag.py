import numpy as np
from qibolab.pulse.pulse_shape.pulse_shape import PulseShape


class Drag(PulseShape):
    """
    Derivative Removal by Adiabatic Gate (DRAG) pulse shape.
    
    Args:
        pulse (Pulse): pulse associated with the shape
        rel_sigma (int): relative sigma so that the pulse standard deviation (sigma) = duration / rel_sigma
    
    .. math::
    
    
    """

    def __init__(self, pulse, rel_sigma, beta):
        self.name = "Drag"
        self.pulse = pulse
        self.rel_sigma = float(rel_sigma)
        self.beta = float(beta)

    @property
    def envelope_i(self):
        x = np.arange(0,self.pulse.duration,1)
        i = self.pulse.amplitude * np.exp(-(1/2)*(((x-(self.pulse.duration-1)/2)**2)/(((self.pulse.duration)/self.rel_sigma)**2)))
        return i

    @property
    def envelope_q(self):
        x = np.arange(0,self.pulse.duration,1)
        i = self.pulse.amplitude * np.exp(-(1/2)*(((x-(self.pulse.duration-1)/2)**2)/(((self.pulse.duration)/self.rel_sigma)**2)))
        q = self.beta * (-(x-(self.pulse.duration-1)/2)/((self.pulse.duration/self.rel_sigma)**2)) * i
        return q

    def __repr__(self):
        return f"{self.name}({self.rel_sigma}, {self.beta})"