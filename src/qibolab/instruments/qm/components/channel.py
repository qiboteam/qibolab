from dataclasses import dataclass

from qibolab.components import Channel

__all__ = [
    "QmChannel",
]


@dataclass(frozen=True)
class QmChannel:
    """Channel for Zurich Instruments (ZI) devices."""

    logical_channel: Channel
    """Corresponding logical channel."""
    device: str
    """Name of the device."""
    port: int
    """Number of port."""
    output: bool = True  # FIXME: Probably not needed
    """Distinguish output from input ports when they have the same numbers."""