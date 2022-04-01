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
                          shape=Gaussian(60 / 5))
    """
    def __init__(self, start, duration, amplitude, frequency, phase, shape, offset_i=0, offset_q=0, channel="qcm", qubit=0):
        # FIXME: Since the ``start`` value depends on the previous pulses we are
        # not sure if it should be a local property of the ``Pulse`` object
        self.start = start
        self.duration = duration
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.shape = shape  # PulseShape objects
        self.channel = channel
        self.offset_i = offset_i
        self.offset_q = offset_q
        self.qubit = qubit

    def serial(self):
        return "P({}, {}, {}, {}, {}, {}, {})".format(self.channel, self.start, self.duration,
                                                      self.amplitude, self.frequency, self.phase, self.shape)

    def compile(self):
        return self.shape.envelope(None, None, self.duration, self.amplitude)

    def __repr__(self):
        return self.serial()


class ReadoutPulse(Pulse):
    """Describes a readout pulse.

    See :class:`qibolab.pulses.Pulse` for argument desciption.
    """

    def __init__(self, start, duration, amplitude, frequency, phase, shape, offset_i=0, offset_q=0, channel="qrm", qubit=0):
        super().__init__(start, duration, amplitude, frequency, phase, shape, offset_i, offset_q, channel, qubit)