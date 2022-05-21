"""Pulse abstractions."""
import numpy as np
from scipy.signal import gaussian
import re
from abc import ABC, abstractmethod
from qibo.config import raise_error


class Pulse:
    """Describes a single pulse to be added to waveform array.

    Args:
        start (float): Start time of pulse in ns.
        duration (float): Pulse duration in ns.
        amplitude (float): Pulse digital amplitude (unitless) [0 to 1].
        frequency (float): Pulse Intermediate Frequency in Hz [10e6 to 300e6].
        phase (float): To be added.
        shape: (str): {'Rectangular', 'Gaussian(rel_sigma)', 'DRAG(rel_sigma, beta)'} Pulse shape.
            See :py:mod:`qibolab.pulses_shapes` for list of available shapes.
        channel (int/str): Specifies the device that will execute this pulse.
        type (str): {'ro', 'qd', 'qf'} type of pulse {ReadOut, Qubit Drive, Qubit Flux}   
        offset_i (float): Optional pulse I offset (unitless).
            (amplitude + offset) should be between [0 and 1].
        offset_q (float): Optional pulse Q offset (unitless).
            (amplitude + offset) should be between [0 and 1].
        qubit (int): qubit associated with the pulse

    Example:
        .. code-block:: python

            from qibolab.pulses import Pulse
            from qibolab.pulse_shapes import Gaussian

            # define pulse with Gaussian shape
            pulse = Pulse(start=0,
                          duration=60,
                          amplitude=0.3,
                          frequency=200000000.0,
                          phase=0,
                          shape=Gaussian(5),
                          channel=1,
                          type='qd')
    """
    def __init__(self, start, duration, amplitude, frequency, phase, shape, channel, type = 'qd', offset_i=0, offset_q=0, qubit=0):
        self.start = start # absolut pulse start time (does not depend on other pulses of the sequence)
        self.duration = duration
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.shape = shape  # (str): {'Rectangular', 'Gaussian(rel_sigma)', 'Drag(rel_sigma, beta)'}
        shape_name = re.findall('(\w+)', shape)[0]
        shape_parameters = re.findall('(\w+)', shape)[1:]
        self.shape_object = globals()[shape_name](self, *shape_parameters) # eval(f"{shape_name}(self, {shape_parameters})")
        self.channel = channel
        self.offset_i = offset_i
        self.offset_q = offset_q
        self.qubit = qubit
        self.type = type # Pulse.type (str): {'qd', 'ro', 'qf'}

    @property
    def serial(self):
        return "P({}, {}, {}, {}, {}, {}, {}, {})".format(self.start, self.duration,
                                                      self.amplitude, self.frequency, self.phase, self.shape, self.channel, self.type)

    @property
    def envelope_i(self):
        return  self.shape_object.envelope_i

    @property
    def envelope_q(self):
        return  self.shape_object.envelope_q

    def __repr__(self):
        return self.serial
 
 
class ReadoutPulse(Pulse):
    """Describes a readout pulse.

    See :class:`qibolab.pulses.Pulse` for argument desciption.
    """

    def __init__(self, start, duration, amplitude, frequency, phase, shape, channel, type = 'ro', offset_i=0, offset_q=0, qubit=0):
        super().__init__(start, duration, amplitude, frequency, phase, shape, channel, type , offset_i, offset_q, qubit)




class PulseShape(ABC):
    """Abstract class for pulse shapes"""
    
    @property
    @abstractmethod
    def envelope_i(self): # pragma: no cover
        raise_error(NotImplementedError)

    @property
    @abstractmethod
    def envelope_q(self): # pragma: no cover
        raise_error(NotImplementedError)


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
        self.name = "gaussian"
        self.pulse = pulse
        self.rel_sigma = rel_sigma 

    @property
    def envelope_i(self):
        return self.pulse.amplitude * gaussian(int(self.pulse.duration), std=int(self.pulse.duration/self.rel_sigma))

    @property
    def envelope_q(self):
        return np.zeros(int(self.pulse.duration))

    def __repr__(self):
        return f"{self.name}({self.rel_sigma})"


class Drag(PulseShape):
    """
    Derivative Removal by Adiabatic Gate (DRAG) pulse shape.
    
    Args:
        pulse (Pulse): pulse associated with the shape
        rel_sigma (int): relative sigma so that the pulse standard deviation (sigma) = duration / rel_sigma
    
    .. math::
    
    
    """

    def __init__(self, rel_sigma, beta):
        self.name = "Drag"
        self.rel_sigma = rel_sigma
        self.beta = beta

    @property
    def envelope_i(self):
        x = np.arange(0,self.pulse.duration,1)
        i = self.pulse.amplitude * np.exp(-(1/2)*(((x-(self.pulse.duration-1)/2)**2)/(((self.pulse.duration)/self.rel_sigma)**2)))
        # same as: self.pulse.amplitude * gaussian(int(self.pulse.duration), std=int(self.pulse.duration/self.rel_sigma))
        return i

    @property
    def envelope_q(self):
        x = np.arange(0,self.pulse.duration,1)
        i = self.pulse.amplitude * np.exp(-(1/2)*(((x-(self.pulse.duration-1)/2)**2)/(((self.pulse.duration)/self.rel_sigma)**2)))
        q = self.beta * (-(x-(self.pulse.duration-1)/2)/((self.pulse.duration/self.rel_sigma)**2)) * i
        return q

    def __repr__(self):
        return f"{self.name}({self.rel_sigma}, {self.beta})"

