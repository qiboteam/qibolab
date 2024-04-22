from dataclasses import dataclass, field, replace
from typing import Dict, Optional

from qibo.config import raise_error

from qibolab.channel_config import (
    AcquisitionChannelConfig,
    DCChannelConfig,
    IQChannelConfig,
)


def check_max_offset(offset, max_offset):
    """Checks if a given offset value exceeds the maximum supported offset.

    This is to avoid sending high currents that could damage lab
    equipment such as amplifiers.
    """
    if max_offset is not None and abs(offset) > max_offset:
        raise_error(
            ValueError, f"{offset} exceeds the maximum allowed offset {max_offset}."
        )


@dataclass
class DCChannel:
    name: str
    config: DCChannelConfig

    max_offset: Optional[float] = None
    """Maximum DC voltage that we can safely send through this channel.

    Sending high voltages for prolonged times may damage amplifiers or
    other lab equipment. If the user attempts to send a higher value an
    error will be raised to prevent execution in real instruments.
    """

    @property
    def offset(self):
        """DC offset that is applied to this port."""
        return self.config.offset

    @offset.setter
    def offset(self, value):
        check_max_offset(value, self.max_offset)
        self.config = replace(self.config, offset=value)


@dataclass
class IQChannel:
    name: str
    config: IQChannelConfig


@dataclass
class AcquisitionChannel:
    name: str
    config: AcquisitionChannelConfig


@dataclass
class ChannelMap:
    """Collection of :class:`qibolab.designs.channel.Channel` objects
    identified by name.

    Essentially, it allows creating a mapping of names to channels just
    specifying the names.
    """

    _channels: Dict[str, Channel] = field(default_factory=dict)

    def add(self, *items):
        """Add multiple items to the channel map.

        If :class: `qibolab.channels.Channel` objects are given they are dded to the channel map.
        If a different type is given, a :class: `qibolab.channels.Channel` with the corresponding name
        is created and added to the channel map.
        """
        for item in items:
            if isinstance(item, Channel):
                self[item.name] = item
            else:
                self[item] = Channel(item)
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
