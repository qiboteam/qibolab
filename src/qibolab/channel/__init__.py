from dataclasses import dataclass, field
from typing import Union

from qibo.config import raise_error

from .configs import *

ChannelConfig = Union[DCChannelConfig, IQChannelConfig, AcquisitionChannelConfig]


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


@dataclass
class ChannelMap:
    """Collection of :class:`Channel` objects identified by name.

    Essentially, it allows creating a mapping of names to channels just
    specifying the names.
    """

    _channels: dict[str, Channel] = field(default_factory=dict)

    def add(self, *items):
        """Add multiple items to the channel map.

        If :class: `qibolab.channels.Channel` objects are given they are dded to the channel map.
        If a different type is given, a :class: `qibolab.channels.Channel` with the corresponding name
        is created and added to the channel map.
        """
        for item in items:
            self[item.name] = item
        return self

    def __getitem__(self, name):
        return self._channels[name]

    def __setitem__(self, name, channel):
        if not isinstance(channel, Channel):
            raise_error(
                TypeError, f"Cannot add channel of type {type(channel)} to ChannelMap."
            )
        self._channels[name] = channel

    def __contains__(self, name):
        return name in self._channels

    def __or__(self, channel_map):
        channels = self._channels.copy()
        channels.update(channel_map._channels)
        return self.__class__(channels)

    def __ior__(self, items):
        if not isinstance(items, type(self)):
            try:
                if isinstance(items, str):
                    raise TypeError
                items = type(self)().add(*items)
            except TypeError:
                items = type(self)().add(items)
        self._channels.update(items._channels)
        return self
