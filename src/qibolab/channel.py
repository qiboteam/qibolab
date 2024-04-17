from dataclasses import dataclass
from enum import Enum, auto

"""
Channel is an abstract concept that defines an interface in front of various instrument drivers in qibolab, without
exposing instrument specific implementation details. For the users of this interface a quantum computer is just a
predefined set of channels where they can send signals or receive signals/data from. Users do not have control over what
channels exist - it is determined by the setup of a certain quantum computer. However, users have control over some
configuration parameters. A typical use case is to configure some channels with desired parameters and request an
execution of a synchronized pulse sequence that implements a certain computation or a calibration experiment.
"""


class ChannelType(Enum):
    DC = auto()
    IQ = auto()
    DIRECT_IQ = auto()
    ACQUISITION = auto()


@dataclass(frozen=True)
class Channel:
    """A channel is represented by its unique name, and the type."""

    name: str
    type: ChannelType
