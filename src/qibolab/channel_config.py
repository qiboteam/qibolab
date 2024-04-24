from dataclasses import dataclass
from typing import Optional, Union

from .execution_parameters import AcquisitionType

"""
Channel is an abstract concept that defines an interface in front of various instrument drivers in qibolab, without
exposing instrument specific implementation details. For the users of this interface a quantum computer is just a
predefined set of channels where they can send signals or receive signals/data from. Users do not have control over what
channels exist - it is determined by the setup of a certain quantum computer. However, users have control over some
configuration parameters. A typical use case is to configure some channels with desired parameters and request an
execution of a synchronized pulse sequence that implements a certain computation or a calibration experiment.
"""


@dataclass(frozen=True)
class DCChannelConfig:
    """Configuration for a channel that can be used to send DC pulses (i.e.
    just envelopes without modulation)."""

    offset: float
    """DC offset/bias of the channel."""


@dataclass(frozen=True)
class LOConfig:
    """Configuration for a local oscillator."""

    frequency: float
    power: float


@dataclass(frozen=True)
class IQMixerConfig:
    """Configuration for IQ mixer.

    Mixers usually have various imperfections, and one needs to
    compensate for them. This class holds the compensation
    configuration.
    """

    offset_i: float = 0.0
    """DC offset for the I component."""
    offset_q: float = 0.0
    """DC offset for the Q component."""
    scale_q: float = 1.0
    """The relative amplitude scale/factor of the q channel, to account for I-Q
    amplitude imbalance."""
    phase_q: float = 0.0
    """The phase offset of the q channel, to account for I-Q phase
    imbalance."""


@dataclass(frozen=True)
class IQChannelConfig:
    """Configuration for an IQ channel."""

    frequency: float
    """The carrier frequency of the channel."""
    lo_config: Optional[LOConfig]
    """Configuration for the corresponding LO.

    None if the channel does not use an LO.
    """
    mixer_config: Optional[IQMixerConfig]
    """Configuration for the corresponding IQ mixer.

    None if the channel does not feature a mixer.
    """


@dataclass(frozen=True)
class AcquisitionChannelConfig:
    """Configuration for acquisition channels."""

    type: AcquisitionType
    twpa_frequency: float
    twpa_power: float


ChannelConfig = Union[DCChannelConfig, IQChannelConfig, AcquisitionChannelConfig]
