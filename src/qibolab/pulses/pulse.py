"""Pulse class."""

from typing import Union

import numpy as np

from qibolab.serialize_ import Model

from .envelope import Envelope, IqWaveform, Waveform

__all__ = [
    "Delay",
    "Pulse",
    "PulseId",
    "PulseLike",
    "VirtualZ",
]

PulseId = int
"""Unique identifier for a pulse."""


class _PulseLike(Model):
    @property
    def id(self) -> PulseId:
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
        """Compute the envelope of the waveform i component."""
        samples = int(self.duration * sampling_rate)
        return self.amplitude * self.envelope.i(samples)

    def q(self, sampling_rate: float) -> Waveform:
        """Compute the envelope of the waveform q component."""
        samples = int(self.duration * sampling_rate)
        return self.amplitude * self.envelope.q(samples)

    def envelopes(self, sampling_rate: float) -> IqWaveform:
        """Compute a tuple with the i and q envelopes."""
        return np.array([self.i(sampling_rate), self.q(sampling_rate)])


class Delay(_PulseLike):
    """Wait instruction.

    For its duration no pulse is sent to the QPU on this channel.
    """

    duration: int
    """Delay duration in ns."""


class VirtualZ(_PulseLike):
    """Implementation of Z-rotations using virtual phase."""

    phase: float
    """Phase that implements the rotation."""

    @property
    def duration(self):
        """Duration of the virtual gate should always be zero."""
        return 0


PulseLike = Union[Pulse, Delay, VirtualZ]
