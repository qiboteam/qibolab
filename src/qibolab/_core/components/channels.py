"""Define channels, representing the physical components handling signals.

Channels are a specific type of component, that are responsible for
generating signals. A channel has a name, and it can refer to the names of
other components as needed. The default configuration of components should be
stored elsewhere (in Platform). By dissecting configuration in smaller pieces
and storing them externally (as opposed to storing the channel configuration
inside the channel itself) has multiple benefits, that.

All revolve around the fact that channels may have shared components, e.g.

- some instruments use one LO for more than one channel,
- for some use cases (like qutrit experiments, or CNOT gates), we need to define multiple channels that point to the same physical wire.

If channels contain their configuration, it becomes cumbersome to make sure that user can easily see that some channels have shared
components, and changing configuration for one may affect the other as well. By storing component configurations externally we
make sure that there is only one copy of configuration for a component, plus users can clearly see when two different channels
share a component, because channels will refer to the same name for the component under discussion.
"""

from typing import Optional

from ..identifier import ChannelId
from ..serialize import Model

__all__ = ["Channel", "DcChannel", "IqChannel", "AcquisitionChannel"]


class Channel(Model):
    """Channel to communicate with the qubit."""

    device: str = ""
    """Name of the device."""
    path: str = ""
    """Physical port addresss within the device."""

    @property
    def port(self) -> int:
        return int(self.path)

    def iqout(self, id_: ChannelId) -> Optional[ChannelId]:
        """Extract associated IQ output channel.

        This is the identity for each IQ output channel identifier, while it retrieves the
        associated probe channel for acquisition ones, and :obj:`None` for any other one
        (essentially, non-RF channels).

        The argument is the identifier of the present channel, since it is not stored within
        the objec itself, as it is only relevant to address it in a collection (and so,
        out of the scope of the object itself).
        """
        return (
            id_
            if isinstance(self, IqChannel)
            else (self.probe if isinstance(self, AcquisitionChannel) else None)
        )


class DcChannel(Channel):
    """Channel that can be used to send DC pulses."""


class IqChannel(Channel):
    """Channel that can be used to send IQ pulses."""

    mixer: Optional[str] = None
    """Name of the IQ mixer component corresponding to this channel.

    None, if the channel does not have a mixer, or it does not need
    configuration.
    """
    lo: Optional[str] = None
    """Name of the local oscillator component corresponding to this channel.

    None, if the channel does not have an LO, or it is not configurable.
    """


class AcquisitionChannel(Channel):
    twpa_pump: Optional[str] = None
    """Name of the TWPA pump component.

    None, if there is no TWPA, or it is not configurable.
    """
    probe: Optional[ChannelId] = None
    """Name of the corresponding measure/probe channel.

    FIXME: This is temporary solution to be able to relate acquisition channel to corresponding probe channel wherever needed in drivers,
    until we make acquire channels completely independent, and users start putting explicit acquisition commands in pulse sequence.
    """
