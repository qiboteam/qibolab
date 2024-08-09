"""Configuration for various components.

These represent the minimal needed configuration that needs to be
exposed to users. Specific definitions of components can expose more,
and that can be used in any troubleshooting or debugging purposes by
users, but in general any user tool should try to depend only on the
configuration defined by these classes.
"""

from dataclasses import dataclass
from typing import Union

__all__ = [
    "DcConfig",
    "IqConfig",
    "AcquisitionConfig",
    "IqMixerConfig",
    "OscillatorConfig",
    "Config",
]


@dataclass(frozen=True)
class DcConfig:
    """Configuration for a channel that can be used to send DC pulses (i.e.
    just envelopes without modulation)."""

    offset: float
    """DC offset/bias of the channel."""


@dataclass(frozen=True)
class OscillatorConfig:
    """Configuration for an oscillator."""

    frequency: float
    power: float


@dataclass(frozen=True)
class IqMixerConfig:
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
class IqConfig:
    """Configuration for an IQ channel."""

    frequency: float
    """The carrier frequency of the channel."""


@dataclass(frozen=True)
class AcquisitionConfig:
    """Configuration for acquisition channel.

    Currently, in qibolab, acquisition channels are FIXME:
    """

    delay: float
    """Delay between readout pulse start and acquisition start."""
    smearing: float
    """FIXME:"""
    threshold: float
    iq_angle: float


Config = Union[DcConfig, IqMixerConfig, OscillatorConfig, IqConfig, AcquisitionConfig]
