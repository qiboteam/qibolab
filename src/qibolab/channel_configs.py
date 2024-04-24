from dataclasses import dataclass
from typing import Union

from .execution_parameters import AcquisitionType

"""Common configuration for various channels."""


@dataclass
class DCChannelConfig:
    """Configuration for a channel that can be used to send DC pulses (i.e.
    just envelopes without modulation on any frequency)"""

    bias: float
    """DC bias/offset of the channel."""


@dataclass
class IQChannelConfig:
    """Configuration for an IQ channel. This is used for IQ channels that can
    generate requested signals by first generating them at an intermediate
    frequency, and then mixing it with a local oscillator (LO) to upconvert to
    the target carrier frequency.

    For this type of IQ channels users typically
        1. want to have control over the LO frequency.
        2. need to be able to calibrate parameters related to the mixer imperfections.
           Mixers typically have some imbalance in the way they treat the I and Q components
           of the signal, and to compensate for it users typically need to calibrate the
           compensation parameters and provide them as channel configuration.
    """

    frequency: float
    """The carrier frequency of the channel."""
    lo_frequency: float
    """The frequency of the local oscillator."""
    mixer_correction_scale: float = 0.0
    """The relative amplitude scale/factor of the q channel."""
    mixer_correction_phase: float = 0.0
    """The phase offset of the q channel of the LO."""


@dataclass
class DirectIQChannelConfig:
    """Configuration for an IQ channel that directly generates signals at
    necessary carrier frequency."""

    frequency: float
    """The carrier frequency of the channel."""


@dataclass
class AcquisitionChannelConfig:
    """Configuration for acquisition channels."""

    type: AcquisitionType


@dataclass
class Channel:
    """A channel is represented by its unique name, and the type of
    configuration that should be specified for it."""

    name: str
    config_type: type


ChannelConfig = Union[
    DCChannelConfig, IQChannelConfig, DirectIQChannelConfig, AcquisitionChannelConfig
]
