from dataclasses import dataclass, field

from qibo.config import raise_error

from .configs import ChannelConfig

__all__ = ["Channel", "ChannelMap"]


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
