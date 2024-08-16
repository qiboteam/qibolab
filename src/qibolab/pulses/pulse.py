"""Pulse class."""

from typing import Union

import numpy as np

from qibolab.serialize import Model

from .envelope import Envelope, IqWaveform, Waveform


class _PulseLike(Model):
    @property
    def id(self) -> int:
        return id(self)


class Pulse(_PulseLike):
    """A pulse to be sent to the QPU."""

    duration: float
    """Pulse duration."""

    amplitude: float
    """Pulse digital amplitude (unitless).

    Pulse amplitudes are normalised between -1 and 1.
    """
    envelope: Envelope
    """The pulse envelope shape.

    See
    :cls:`qibolab.pulses.envelope.Envelopes` for list of available shapes.
    """
    relative_phase: float = 0.0
    """Relative phase of the pulse, in radians."""

    @classmethod
    def flux(cls, **kwargs):
        """Construct a flux pulse.

        It provides a simplified syntax for the :cls:`Pulse` constructor, by applying
        suitable defaults.
        """
        kwargs["relative_phase"] = 0
        return cls(**kwargs)

    def i(self, sampling_rate: float) -> Waveform:
        """The envelope waveform of the i component of the pulse."""
        samples = int(self.duration * sampling_rate)
        return self.amplitude * self.envelope.i(samples)

    def q(self, sampling_rate: float) -> Waveform:
        """The envelope waveform of the q component of the pulse."""
        samples = int(self.duration * sampling_rate)
        return self.amplitude * self.envelope.q(samples)

    def envelopes(self, sampling_rate: float) -> IqWaveform:
        """A tuple with the i and q envelope waveforms of the pulse."""
        return np.array([self.i(sampling_rate), self.q(sampling_rate)])


class Delay(_PulseLike):
    """Wait instruction.

    During its length no pulse is sent on the same channel.
    """

    duration: float
    """Duration in ns."""


class VirtualZ(_PulseLike):
    """Implementation of Z-rotations using virtual phase."""

    phase: float
    """Phase that implements the rotation."""

    @property
    def duration(self):
        """Duration of the virtual gate should always be zero."""
        return 0


class Acquisition(_PulseLike):
    """Acquisition instruction.

    This event instructs the device to acquire samples for the event
    span.
    """

    duration: float
    """Duration in ns."""


PulseLike = Union[Pulse, Delay, VirtualZ, Acquisition]
