import numpy as np
from qibolab.pulse.pulse_shape.pulse_shape import PulseShape


class Rectangular(PulseShape):
    """
    Rectangular pulse shape.
    
    Args:
        pulse (Pulse): pulse associated with the shape
    """
    def __init__(self, pulse):
        self.name = "Rectangular"
        self.pulse = pulse

    @property
    def envelope_i(self):
        return self.pulse.amplitude * np.ones(int(self.pulse.duration))

    @property
    def envelope_q(self):
        return np.zeros(int(self.pulse.duration))

    def __repr__(self):
        return f"{self.name}()"