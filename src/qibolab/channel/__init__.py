from dataclasses import dataclass
from typing import Union

from .configs import *

ChannelConfig = Union[DcConfig, IqConfig, AcquisitionConfig]


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

    twpa_pump: str
    """The name of the oscillator instrument used as TWPA pump."""


def external_config(channel: Channel) -> dict[str, str]:
    """Identifies which parts of the configuration of given channel should be
    used to configure and externally connected supplementary instrument.

    Returns:
        Dictionary mapping config attribute to instrument name.
    """

    if isinstance(channel, WithExternalLo):
        return {"lo": channel.lo}
    if isinstance(channel, WithExternalTwpaPump):
        return {"twpa_pump": channel.twpa_pump}
    return {}
