from dataclasses import dataclass, field
from typing import Optional, Union

from .configs import (
    AcquisitionConfig,
    AuxIqConfig,
    DcConfig,
    IqConfig,
    OscillatorConfig,
)

__all__ = [
    "ChannelConfig",
    "Channel",
    "WithExternalLo",
    "WithExternalTwpaPump",
    "external_config",
]


ChannelConfig = Union[AcquisitionConfig, DcConfig, IqConfig]
AuxChannelConfig = Union[AuxIqConfig]

AUX_SUFFIX = "__aux"


@dataclass(frozen=True)
class Channel:
    """Channel is an abstract concept that defines means of communication
    between users and a quantum computer.

    A quantum computer can be perceived as just a set of channels where
    signals can be sent to or received from. Channels are identified
    with a unique name. The type of channel is inferred from the type of
    config it accepts.
    """

    name: str
    config: ChannelConfig

    _aux_channels: dict[str, AuxChannelConfig] = field(init=False, default_factory=dict)

    def __post_init__(self):
        if AUX_SUFFIX in self.name:
            raise ValueError(f"Channel name must not contain {AUX_SUFFIX}.")

    @property
    def aux_channels(self) -> dict[str, AuxChannelConfig]:
        return self._aux_channels.copy()

    def create_aux_channel(self, config: AuxChannelConfig) -> str:
        if isinstance(self.config, IqConfig) and not isinstance(config, AuxIqConfig):
            raise ValueError(
                f"Channel {self.name} with configuration of type {IqConfig} (IQ channel) cannot contain aux channel "
                f"with configuration of type {type(config)}"
            )
        elif not isinstance(self.config, IqConfig):
            raise ValueError(f"Aux channels are supported for IQ channels only.")
        name = f"{self.name}{AUX_SUFFIX}_{len(self._aux_channels)}"
        self._aux_channels[name] = config
        return name


@dataclass(frozen=True)
class WithExternalLo:
    """Mixin class to be used for instrument specific IQ channel definitions,
    in case the instrument does not have internal up-conversion unit and relies
    on an external local oscillator (LO)."""

    lo: str
    """The name of the external local oscillator instrument."""


@dataclass(frozen=True)
class WithExternalTwpaPump:
    """Mixin class to be used for instrument specific acquisition channel
    definitions, in case the instrument does not have built-in oscillator
    dedicated as TWPA pump and an external TWPA pump should be used."""

    twpa_pump: Optional[str]
    """The name of the oscillator instrument used as TWPA pump.

    None, if the TWPA pump/TWPA is not installed in the setup.
    """


def external_config(
    channel: Channel, config: Optional[ChannelConfig] = None
) -> dict[str, OscillatorConfig]:
    """Extracts parts of given channel's configuration that should be used to
    configure and externally connected supplementary instrument. If config
    argument is provided it will take precedence over the config available
    inside the channel object.

    Returns:
        Dictionary mapping instrument name to corresponding extracted config.
    """
    cfg = config or channel.config
    if isinstance(channel, WithExternalLo):
        return {channel.lo: cfg.lo_config}
    if isinstance(channel, WithExternalTwpaPump):
        return {channel.twpa_pump: cfg.twpa_pump_config}
    return {}
