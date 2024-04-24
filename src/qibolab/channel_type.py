from typing import Protocol

from .channel_config import ChannelConfig


class Channel(Protocol):
    name: str
    config: ChannelConfig
