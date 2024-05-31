from dataclasses import dataclass
from typing import Optional, Union

from .configs import AcquisitionConfig, DcConfig, IqConfig, OscillatorConfig

__all__ = [
    "ChannelConfig",
    "Channel",
    "WithExternalLo",
    "WithExternalTwpaPump",
    "external_config",
]


ChannelConfig = Union[AcquisitionConfig, DcConfig, IqConfig]


@dataclass(frozen=True)
class Channel:
    """Channel is an abstract concept that defines means of communication
    between users and a quantum computer.

    A quantum computer can be perceived as just a set of channels where
    signals can be sent to or received from. Channels are identified
    with a unique name. The type of a channel is inferred from the type
    of config it accepts.
    """

    name: str
    config: ChannelConfig


@dataclass(frozen=True)
class WithExternalLo:
    """Mixin class to be used for instrument specific IQ channel definitions,
    in case the instrument does not have internal up-conversion unit and relies
    on an external local oscillator (LO)."""

    lo: str
    """The name of the external local oscillator instrument."""


@dataclass(frozen=True)
class WithExternalTwpaPump:
    """Mixin class to be used for instrument specific acquisition channel
    definitions, in case the instrument does not have built-in oscillator
    dedicated as TWPA pump and an external TWPA pump should be used."""

    twpa_pump: Optional[str]
    """The name of the oscillator instrument used as TWPA pump.

    None, if the TWPA pump/TWPA is not installed in the setup.
    """


def external_config(
    channel: Channel, config: Optional[ChannelConfig] = None
) -> dict[str, OscillatorConfig]:
    """Extracts parts of given channel's configuration that should be used to
    configure and externally connected supplementary instrument. If config
    argument is provided it will take precedence over the config available
    inside the channel object.

    Returns:
        Dictionary mapping instrument name to corresponding extracted config.
    """
    cfg = config or channel.config
    if isinstance(channel, WithExternalLo):
        return {channel.lo: cfg.lo_config}
    if isinstance(channel, WithExternalTwpaPump):
        return {channel.twpa_pump: cfg.twpa_pump_config}
    return {}
