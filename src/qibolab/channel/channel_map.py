from dataclasses import dataclass, field

from qibo.config import raise_error

from . import Channel

__all__ = ["ChannelMap"]


@dataclass
class ChannelMap:
    """Collection of :class:`qibolab.designs.channel.Channel` objects
    identified by name.

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
