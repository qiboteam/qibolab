from dataclasses import dataclass
from typing import Optional

"""Common configuration for various channels."""

__all__ = [
    "DcConfig",
    "IqConfig",
    "AcquisitionConfig",
    "OscillatorConfig",
    "IqMixerConfig",
]


@dataclass(frozen=True)
class DcConfig:
    """Configuration for a channel that can be used to send DC pulses (i.e.
    just envelopes without modulation)."""

    offset: float
    """DC offset/bias of the channel."""


@dataclass(frozen=True)
class OscillatorConfig:
    """Configuration for a local oscillator."""

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

    lo_config: Optional[OscillatorConfig]
    """Configuration for the corresponding LO.

    None if the channel does not use an LO.
    """
    mixer_config: Optional[IqMixerConfig]
    """Configuration for the corresponding IQ mixer.

    None if the channel does not feature a mixer.
    """


@dataclass(frozen=True)
class AcquisitionConfig:
    """Configuration for acquisition channels."""

    twpa_pump_config: Optional[OscillatorConfig]
    """Config for the corresponding TWPA pump.

    None if the channel does not feature a TWPA.
    """
