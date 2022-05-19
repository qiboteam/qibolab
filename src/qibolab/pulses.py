"""Pulse abstractions."""
import bisect
import numpy as np


class Pulse:
    """Describes a single pulse to be added to waveform array.

    Args:
        start (float): Start time of pulse in ns.
        duration (float): Pulse duration in ns.
        amplitude (float): Pulse digital amplitude (unitless) [0 to 1].
        frequency (float): Pulse Intermediate Frequency in Hz [10e6 to 300e6].
        phase (float): To be added.
        shape: (PulseShape): Pulse shape.
            See :py:mod:`qibolab.pulses_shapes` for list of available shapes.
        offset_i (float): Optional pulse I offset (unitless).
            (amplitude + offset) should be between [0 and 1].
        offset_q (float): Optional pulse Q offset (unitless).
            (amplitude + offset) should be between [0 and 1].
        channel (int/str): Specifies the device that will execute this pulse.
            FPGA channel (int) for IcarusQ or qrm/qcm (str) for TIIq.
        qubit (int): Target qubit ID

    Example:
        .. code-block:: python

            from qibolab.pulses import Pulse
            from qibolab.pulse_shapes import Gaussian

            # define pulse with Gaussian shape
            pulse = Pulse(start=0,
                          frequency=200000000.0,
                          amplitude=0.3,
                          duration=60,
                          phase=0,
                          shape=Gaussian(5))
    """
    def __init__(self, start, duration, amplitude, frequency, phase, shape, channel, type = 'qd', offset_i=0, offset_q=0, qubit=0):
        self.start = start # absolut pulse start time (does not depend on other pulses of the sequence)
        self.duration = duration
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.shape = shape  # (str): {'Rectangular', 'Gaussian(rel_sigma)', 'DRAG(rel_sigma, beta)'}
        self.channel = channel
        self.offset_i = offset_i
        self.offset_q = offset_q
        self.qubit = qubit
        self.type = type # Pulse.type (str): {'qd', 'ro', 'qf'}

    @property
    def serial(self):
        return "P({}, {}, {}, {}, {}, {}, {}, {})".format(self.channel, self.start, self.duration,
                                                      self.amplitude, self.frequency, self.phase, self.shape, self.type)

    def compile(self):
        return self.shape.envelope(None, None, self.duration, self.amplitude)

    @property
    def shape_parameters(self):
        shape = str(self.shape)
        parameters = []
        if '(' in shape:
            if ')' in shape[shape.find('(')+1:]:
                shape = shape[shape.find('(')+1:shape.find(')')]
                if len(shape)>0:
                    parameters = [parameter.strip() for parameter in shape.split(',')]
        return parameters


    @property
    def envelope_i(self):
        if 'Rectangular' in self.shape:
            envelope = self.amplitude * np.ones(int(self.duration))
        elif 'Gaussian' in self.shape:
            """Gaussian envelope centered with respect to the pulse.
            Gaussian(rel_sigma)
            example: Gaussian(5)

            .. math::

                A\exp^{-\\frac{1}{2}\\frac{(t-\mu)^2}{\sigma^2}}

                where sigma = duration/rel_sigma
            """
            from scipy.signal import gaussian
            rel_sigma = int(self.shape_parameters[0])
            envelope =  self.amplitude * gaussian(int(self.duration), std=int(self.duration/rel_sigma))

        elif 'DRAG' in self.shape:
            """DRAG envelope centered with respect to the pulse.
            DRAG(rel_sigma, beta)
            example: DRAG(5,1)
            .. math::
                G + i\\beta(-\\frac{t-\mu}{\sigma^2})G

            .. math::
                G = A\exp^{-\\frac{1}{2}\\frac{(t-\mu)^2}{\sigma^2}}

                where sigma = duration/rel_sigma

            """
            """
            from scipy.signal import gaussian
            rel_sigma = int(self.shape_parameters[0])
            envelope =  self.amplitude * gaussian(int(self.duration), std=int(self.duration/rel_sigma))
            """
            raise NotImplementedError

        elif 'SWIPHT' in self.shape:
            raise NotImplementedError
        else:
            raise NotImplementedError
            
        return envelope
    @property
    def envelope_q(self):
        if 'Rectangular' in self.shape:
            envelope = np.zeros(int(self.duration)) 
        elif 'Gaussian' in self.shape:
            envelope =  np.zeros(int(self.duration)) 
        elif 'DRAG' in self.shape:
            # FIXME: Fix implementation
            """
            rel_sigma = self.shape_parameters[0]
            beta = self.shape_parameters[1]
            
            mu = self.duration / 2
            gaussian = self.amplitude * np.exp(-0.5 * (time - mu) ** 2 / rel_sigma ** 2)
            envelope = gaussian + 1j * beta * (-(time - mu) / rel_sigma ** 2) * gaussian
            """
            raise NotImplementedError
        elif 'SWIPHT' in self.shape:
            raise NotImplementedError
        else:
            raise NotImplementedError
        return envelope

    def __repr__(self):
        return self.serial
 
 
class ReadoutPulse(Pulse):
    """Describes a readout pulse.

    See :class:`qibolab.pulses.Pulse` for argument desciption.
    """

    def __init__(self, start, duration, amplitude, frequency, phase, shape, channel, type = 'ro', offset_i=0, offset_q=0, qubit=0):
        super().__init__(start, duration, amplitude, frequency, phase, shape, channel, type , offset_i, offset_q, qubit)