from typing import Optional

from pydantic import ConfigDict, Field

# TODO: the unused import are there because Qibocal is still importing them from here
# since the export scheme will be reviewed, it should be changed at that time, removing
# the unused ones from here
from .identifier import ChannelId, QubitId, QubitPairId, TransitionId  # noqa
from .serialize import Model


class Qubit(Model):
    """Representation of a physical qubit.

    Contains the channel ids used to control the qubit and is instantiated
    in the function that creates the corresponding
    :class:`qibolab.platforms.platform.Platform`
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

    @classmethod
    def default(cls, name: QubitId, channels: Optional[list[str]] = None, **kwargs):
        """Create a qubit with default channel names.

        Default channel names follow the convention:
        '{qubit_name}/{channel_type}'

        Args:
            name: Name for the qubit to be used for channel ids.
            channels: List of channels to add to the qubit.
                If ``None`` the following channels will be added:
                probe, acquisition, drive and flux.
        """
        if channels is None:
            channels = ["probe", "acquisition", "drive", "flux"]
        return cls(**{ch: f"{name}/{ch}" for ch in channels}, **kwargs)


class QubitPair(Model):
    """Represent a two-qubit interaction."""

    drive: Optional[ChannelId] = None
    """Output channel, for cross-resonance driving."""
