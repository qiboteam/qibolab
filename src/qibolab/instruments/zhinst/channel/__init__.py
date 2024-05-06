from dataclasses import dataclass

from qibolab.channel import Channel
from qibolab.instruments.oscillator import LocalOscillator

from .configs import *


@dataclass(frozen=True)
class ZIChannel(Channel):

    device: str
    path: str


@dataclass(frozen=True)
class ZIAcquisitionChannel(ZIChannel):

    twpa_pump: LocalOscillator
