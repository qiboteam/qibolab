from dataclasses import dataclass, field
from typing import List, Optional

from qibo.config import raise_error

from qibolab.instruments.abstract import LocalOscillator


@dataclass
class Channel:
    """Representation of physical wire connection (channel)."""

    name: str
    """Name of the channel from the lab schematics."""

    qubit: Optional["Qubit"] = field(default=None, repr=False)
    """Qubit connected to this channel.
    Used to read the sweetspot and filters for flux channels only. ``None`` for non-flux channels.
    """
    ports: List[tuple] = field(default_factory=list)
    """List of tuples (controller, port) connected to this channel."""
    local_oscillator: Optional[LocalOscillator] = None
    """Instrument object controlling the local oscillator connected to this channel.
    Not applicable for setups that do not use local oscillators because the controller
    can send sufficiently high frequencies
    """
    _bias: Optional[float] = None
    """DC offset that should be applied in the channel in order to shift the
    frequency of the qubit, usually to put it in its sweetspot.
    Relevant only for flux channels and flux-tunable transmon qubits.
    """
    _filter: Optional[dict] = None
    """Filter to be applied to the channel to reduce the distortions when sending
    flux pulses. Useful for two-qubit gates.
    Quantum Machines associate filters to channels but this may not be the case
    in other instruments.
    """

    @property
    def bias(self):
        """Bias offset for flux channels."""
        if self._bias is None:
            # operate qubits at their sweetspot unless otherwise stated
            return self.qubit.sweetspot
        return self._bias

    @bias.setter
    def bias(self, bias):
        if not isinstance(bias, (int, float)):
            raise_error(TypeError, f"Attempting to set non-float bias {bias}.")
        self._bias = bias

    @property
    def filter(self):
        """Filters for sending flux pulses through a flux channel."""
        if self._filter is None:
            return self.qubit.filter
        return self._filter

    @filter.setter
    def filter(self, filter):
        if not isinstance(filter, dict):
            raise_error(TypeError, f"Channel filter must be dict but is {type(filter)}.")
        self._filter = filter


@dataclass
class ChannelMap:
    """Collection of :class:`qibolab.designs.channel.Channel` objects identified by name."""

    _channels: dict = field(default_factory=dict)

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
        self._channels.update(channel_map._channels)
        return self
