from dataclasses import dataclass

from qibolab.components import Channel

__all__ = [
    "QmChannel",
]


@dataclass(frozen=True)
class QmChannel:
    """Channel for Quantum Machines devices."""

    logical_channel: Channel
    """Corresponding logical channel."""
    device: str
    """Name of the device."""
    port: int
    """Number of port."""
