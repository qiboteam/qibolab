from dataclasses import dataclass

from qibolab.channel import Channel, WithExternalTwpaPump

from .configs import *


@dataclass(frozen=True)
class ZIChannel(Channel):
    """Channel for Zurich Instruments (ZI) devices."""

    device: str
    """Name of the device."""
    path: str
    """Path of the device node."""


@dataclass(frozen=True)
class ZiAcquisitionChannel(WithExternalTwpaPump, ZIChannel): ...
