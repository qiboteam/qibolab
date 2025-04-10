from typing import Annotated, Optional, Union

from pydantic import ConfigDict, Field

from .identifier import ChannelId, QubitId, TransitionId
from .serialize import Model

__all__ = ["Qubit"]

DefaultChannelType = Annotated[Optional[ChannelId], True]
"""If ``True`` the channel is included in the default qubit constructor."""


class Qubit(Model):
    """Representation of a physical qubit.

    Contains the channel ids used to control the qubit and is instantiated
    in the function that creates the corresponding
    :class:`qibolab.Platform`
    """

    model_config = ConfigDict(frozen=False)

    drive: DefaultChannelType = None
    """Ouput channel, to drive the qubit state."""
    drive_extra: Annotated[dict[Union[TransitionId, QubitId], ChannelId], False] = (
        Field(default_factory=dict)
    )
    """Output channels collection, to drive non-qubit transitions."""
    flux: DefaultChannelType = None
    """Output channel, to control the qubit flux."""
    probe: DefaultChannelType = None
    """Output channel, to probe the resonator."""
    acquisition: DefaultChannelType = None
    """Input channel, to acquire the readout results."""

    @property
    def channels(self) -> list[ChannelId]:
        return [
            x
            for x in (
                [getattr(self, ch) for ch in ["probe", "acquisition", "drive", "flux"]]
                + list(self.drive_extra.values())
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
            channels = [name for name, f in cls.model_fields.items() if f.metadata[0]]
        return cls(**{ch: f"{name}/{ch}" for ch in channels}, **kwargs)
