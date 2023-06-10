from dataclasses import dataclass
from typing import Optional

from qibo.config import raise_error

from qibolab.instruments.oscillator import LocalOscillator
from qibolab.instruments.port import Port
from qibolab.maps import NamedMap, NamedType


def check_max_offset(offset, max_offset):
    """Checks if a given offset value exceeds the maximum supported offset.

    This is to avoid sending high currents that could damage lab equipment
    such as amplifiers.
    """
    if max_offset is not None and abs(offset) > max_offset:
        raise_error(ValueError, f"{offset} exceeds the maximum allowed offset {max_offset}.")


@dataclass
class Channel(NamedType):
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


class ChannelMap(NamedMap):
    Type = Channel
