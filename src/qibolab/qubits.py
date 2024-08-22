from typing import Optional

from pydantic import ConfigDict, Field

from .identifier import ChannelId, TransitionId
from .serialize import Model


class Qubit(Model):
    """Representation of a physical qubit.

    Qubit objects are instantiated by :class:`qibolab.platforms.platform.Platform`
    but they are passed to instrument designs in order to play pulses.
    """

    model_config = ConfigDict(frozen=False)

    drive: Optional[ChannelId] = None
    """Ouput channel, to drive the qubit state."""
    drive_qudits: dict[TransitionId, ChannelId] = Field(default_factory=dict)
    """Output channels collection, to drive non-qubit transitions."""
    flux: Optional[ChannelId] = None
    """Output channel, to control the qubit flux."""
    probe: Optional[ChannelId] = None
    """Output channel, to probe the resonator."""
    acquisition: Optional[ChannelId] = None
    """Input channel, to acquire the readout results."""

    @property
    def channels(self) -> list[ChannelId]:
        return [
            x
            for x in (
                [getattr(self, ch) for ch in ["probe", "acquisition", "drive", "flux"]]
                + list(self.drive_qudits.values())
            )
            if x is not None
        ]


class QubitPair(Model):
    """Represent a two-qubit interaction."""

    drive: Optional[ChannelId] = None
    """Output channel, for cross-resonance driving."""
