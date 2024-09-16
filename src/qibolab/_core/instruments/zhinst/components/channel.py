from dataclasses import dataclass

from qibolab.components import Channel

__all__ = [
    "ZiChannel",
]


@dataclass(frozen=True)
class ZiChannel:
    """Channel for Zurich Instruments (ZI) devices."""

    logical_channel: Channel
    """Corresponding logical channel."""
    device: str
    """Name of the device."""
    path: str
    """Path of the device node."""
