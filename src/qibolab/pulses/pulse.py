"""Pulse class."""

from typing import Annotated, Literal, Union

import numpy as np
from pydantic import Field

from qibolab.serialize import Model

from .envelope import Envelope, IqWaveform, Waveform


class _PulseLike(Model):
    @property
    def id(self) -> int:
        """Instruction identifier."""
        return id(self)


class Pulse(_PulseLike):
    """A pulse to be sent to the QPU.

    Valid on any channel, except acquisition ones.
    """

    kind: Literal["pulse"] = "pulse"

    duration: float
    """Pulse duration."""

    amplitude: float
    """Pulse digital amplitude (unitless).

    Pulse amplitudes are normalised between -1 and 1.
    """
    envelope: Envelope
    """The pulse envelope shape.

    See
    :class:`qibolab.pulses.envelope.Envelopes` for list of available shapes.
    """
    relative_phase: float = 0.0
    """Relative phase of the pulse, in radians."""

    @classmethod
    def flux(cls, **kwargs):
        """Construct a flux pulse.

        It provides a simplified syntax for the :class:`Pulse` constructor, by applying
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

    Valid on any channel.
    """

    kind: Literal["delay"] = "delay"

    duration: float
    """Duration in ns."""


class VirtualZ(_PulseLike):
    """Implementation of Z-rotations using virtual phase.

    Only valid on a drive channel.
    """

    kind: Literal["virtualz"] = "virtualz"

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

    Only valid on an acquisition channel.
    """

    kind: Literal["acquisition"] = "acquisition"

    duration: float
    """Duration in ns."""


class _Readout(_PulseLike):
    """Readout instruction.

    This event instructs the device to acquire samples for the event
    span.

    Only valid on an acquisition channel.
    """

    kind: Literal["readout"] = "readout"

    acquisition: Acquisition
    probe: Pulse

    @property
    def duration(self) -> float:
        """Duration in ns."""
        return self.acquisition.duration

    @property
    def id(self) -> int:
        """Instruction identifier."""
        return self.acquisition.id


class Align(_PulseLike):
    """Brings different channels at the same point in time."""


PulseLike = Annotated[
    Union[Align, Pulse, Delay, VirtualZ, Acquisition, _Readout], Field(discriminator="kind")
]
