from dataclasses import dataclass
from typing import Union

from .channel_map import *
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
