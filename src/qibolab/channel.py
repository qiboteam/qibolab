from dataclasses import dataclass

from .channel_config import ChannelConfig


@dataclass(frozen=True)
class Channel:
    name: str
    config: ChannelConfig
