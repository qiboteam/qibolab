"""Channels are a specific type of component, that are responsible for
generating signals. A channel has a name, and it can refer to the names of
other components as needed. The default configuration of components should be
stored elsewhere (in Platform). By dissecting configuration in smaller pieces
and storing them externally (as opposed to storing the channel configuration
inside the channel itself) has multiple benefits, that.

all revolve around the fact that channels may have shared components, e.g.
 - Some instruments use one LO for more than one channel,
 - For some use cases (like qutrit experiments, or CNOT gates), we need to define multiple channels that point to the same physical wire.
If channels contain their configuration, it becomes cumbersome to make sure that user can easily see that some channels have shared
components, and changing configuration for one may affect the other as well. By storing component configurations externally we
make sure that there is only one copy of configuration for a component, plus users can clearly see when two different channels
share a component, because channels will refer to the same name for the component under discussion.
"""

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "Channel",
    "DcChannel",
    "IqChannel",
    "AcquireChannel",
]


@dataclass(frozen=True)
class Channel:
    name: str
    """Name of the channel."""


@dataclass(frozen=True)
class DcChannel(Channel):
    """Channel that can be used to send DC pulses."""


@dataclass(frozen=True)
class IqChannel(Channel):
    """Channel that can be used to send IQ pulses."""

    mixer: Optional[str]
    """Name of the IQ mixer component corresponding to this channel.

    None, if the channel does not have a mixer, or it does not need
    configuration.
    """
    lo: Optional[str]
    """Name of the local oscillator component corresponding to this channel.

    None, if the channel does not have an LO, or it is not configurable.
    """
    acquisition: Optional[str] = None
    """In case self is a readout channel this shall contain the name of the
    corresponding acquire channel.

    FIXME: This is temporary solution to be able to generate acquisition commands on correct channel in drivers,
    until we make acquire channels completely independent, and users start putting explicit acquisition commands in pulse sequence.
    """


@dataclass(frozen=True)
class AcquireChannel(Channel):
    twpa_pump: Optional[str]
    """Name of the TWPA pump component.

    None, if there is no TWPA, or it is not configurable.
    """
    probe: Optional[str] = None
    """Name of the corresponding measure/probe channel.

    FIXME: This is temporary solution to be able to relate acquisition channel to corresponding probe channel wherever needed in drivers,
    until we make acquire channels completely independent, and users start putting explicit acquisition commands in pulse sequence.
    """
