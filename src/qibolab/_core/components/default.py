from .channels import AcquisitionChannel, Channel, DcChannel, IqChannel
from .configs import AcquisitionConfig, Config, DcConfig, IqConfig

__all__ = ["channel_to_config"]

CHANNEL_TO_CONFIG_MAP = {
    Channel: Config,
    DcChannel: lambda: DcConfig(offset=0),
    IqChannel: lambda: IqConfig(frequency=0),
    AcquisitionChannel: lambda: AcquisitionConfig(delay=0, smearing=0),
}


def channel_to_config(channel: Channel) -> Config:
    """Create a default config for a given channel.

    The config type depends on the channel type.
    """
    return CHANNEL_TO_CONFIG_MAP[type(channel)]()
