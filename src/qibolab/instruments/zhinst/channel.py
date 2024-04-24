from dataclasses import dataclass

from qibolab.channel import Channel


@dataclass(frozen=True)
class ZIChannel(Channel):

    device: str
    path: str
