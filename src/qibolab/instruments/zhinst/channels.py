from dataclasses import dataclass

from qibolab.channel_config import (
    AcquisitionChannelConfig,
    DCChannelConfig,
    IQChannelConfig,
)
from qibolab.instruments.oscillator import LocalOscillator


@dataclass(frozen=True)
class DCChannelConfigZI(DCChannelConfig):
    """DC channel config using ZI HDAWG."""

    power_range: float
    """Power range in volts."""


@dataclass(frozen=True)
class IQChannelConfigZI(IQChannelConfig):
    """IQ channel config for ZI SHF* line instrument."""

    power_range: int
    """Db"""


class ZIChannel:
    name: str
    device: str
    path: str


class DCChannelZI(ZIChannel):
    config: DCChannelConfigZI


class IQChannelZI(ZIChannel):
    config: IQChannelConfigZI
    lo: LocalOscillator


class AcquisitionChannelZI(ZIChannel):
    config: AcquisitionChannelConfig
