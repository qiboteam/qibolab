from dataclasses import dataclass, field
from typing import Dict, List, Optional

from qibo.config import raise_error

from qibolab.instruments.oscillator import LocalOscillator
from qibolab.instruments.port import Port


def check_max_offset(offset, max_offset):
    """Checks if a given offset value exceeds the maximum supported offset.

    This is to avoid sending high currents that could damage lab equipment
    such as amplifiers.
    """
    if max_offset is not None and abs(offset) > max_offset:
        raise_error(ValueError, f"{offset} exceeds the maximum allowed offset {max_offset}.")


@dataclass
class Channel:
    """Representation of physical wire connection (channel)."""

    name: str
    """Name of the channel from the lab schematics."""
    port: Optional[Port] = None
    """Instrument port that is connected to this channel."""
    local_oscillator: Optional[LocalOscillator] = None
    """Instrument object controlling the local oscillator connected to this channel.

    Not applicable for setups that do not use external local oscillators because the
    controller can send sufficiently high frequencies or contains internal local
    oscillators.
    """
    max_offset: Optional[float] = None
    """Maximum DC voltage that we can safely send through this channel.

    Sending high voltages for prolonged times may damage amplifiers or other lab equipment.
    If the user attempts to send a higher value an error will be raised to prevent
    execution in real instruments.
    """

    @property
    def offset(self):
        """DC offset that is applied to this port."""
        return self.port.offset

    @offset.setter
    def offset(self, value):
        check_max_offset(value, self.max_offset)
        self.port.offset = value

    @property
    def lo_frequency(self):
        if self.local_oscillator is not None:
            return self.local_oscillator.frequency
        return self.port.lo_frequency

    @lo_frequency.setter
    def lo_frequency(self, value):
        if self.local_oscillator is not None:
            self.local_oscillator.frequency = value
        else:
            self.port.lo_frequency = value

    @property
    def lo_power(self):
        if self.local_oscillator is not None:
            return self.local_oscillator.power
        return self.port.lo_power

    @lo_power.setter
    def lo_power(self, value):
        if self.local_oscillator is not None:
            self.local_oscillator.power = value
        else:
            self.port.lo_power = value

    # TODO: gain, attenuation and power range can be unified to a single property
    @property
    def gain(self):
        return self.port.gain

    @gain.setter
    def gain(self, value):
        self.port.gain = value

    @property
    def attenuation(self):
        """Attenuation that is applied to this port."""
        raise_error(NotImplementedError, "Instruments do not support attenuation.")

    @attenuation.setter
    def attenuation(self, value):
        self.port.attenuation = value

    @property
    def power_range(self):
        return self.port.power_range

    @power_range.setter
    def power_range(self, value):
        self.port.power_range = value

    @property
    def filter(self):
        return self.port.filter

    @filter.setter
    def filter(self, value):
        self.port.filter = value


@dataclass
class ChannelMap:
    """Collection of :class:`qibolab.designs.channel.Channel` objects identified by name.

    Essentially, it allows creating a mapping of names to channels just
    specifying the names.

    """

    _channels: Dict[str, Channel] = field(default_factory=dict)

    @classmethod
    def from_names(cls, *names):
        """Construct multiple :class:`qibolab.designs.channel.Channel` objects from names.

        Args:
            names (str): List of channel names.
        """
        return cls({name: Channel(name) for name in names})

    def __getitem__(self, name):
        return self._channels[name]

    def __setitem__(self, name, channel):
        if not isinstance(channel, Channel):
            raise_error(TypeError, f"Cannot add channel of type {type(channel)} to ChannelMap.")
        self._channels[name] = channel

    def __contains__(self, name):
        return name in self._channels

    def __or__(self, channel_map):
        channels = self._channels.copy()
        channels.update(channel_map._channels)
        return self.__class__(channels)

    def __ior__(self, channel_map):
        if not isinstance(channel_map, type(self)):
            try:
                if isinstance(channel_map, str):
                    raise TypeError
                channel_map = type(self).from_names(*channel_map)
            except TypeError:
                channel_map = type(self).from_names(channel_map)
        self._channels.update(channel_map._channels)
        return self
