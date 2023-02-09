
""" RFSoC fpga driver.
Supports the following FPGA:
    RFSoC 4x2
"""
import json
import socket

import numpy as np
import yaml
from qibo.config import log

from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.pulses import (
    Drag,
    Gaussian,
    Pulse,
    PulseSequence,
    PulseShape,
    ReadoutPulse,
    Rectangular,
)
class TII_RFSOC4x2():
    def __init__(self, name: str, address: str):  # , setting_parameters: dict):
        super().__init__(name, address)
        self.cfg: dict = {}
        self.host: str
        self.port: str
        self.host, port = address.split(":")
        self.port = int(port)