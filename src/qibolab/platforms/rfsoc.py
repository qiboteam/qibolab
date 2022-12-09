import json
import socket

import numpy as np
import yaml
from qibo.config import log

from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import (
    Drag,
    Gaussian,
    Pulse,
    PulseSequence,
    ReadoutPulse,
    Rectangular,
)


class RFSoc1qPlatform(AbstractPlatform):
    def __init__(self, name, runcard):
        log.info(f"Loading platform {name} from runcard {runcard}")
        self.name = name
        self.runcard = runcard
        # self.is_connected = False
        # Load platform settings
        with open(runcard) as file:
            self.settings = yaml.safe_load(file)
        #        address =   self.settings["instruments"]["tii_rfsoc4x2"]["settings"]["ip_address"]
        lib = self.settings["instruments"][name]["lib"]
        i_class = self.settings["instruments"][name]["class"]
        address = self.settings["instruments"][name]["settings"]["ip_address"]
        from importlib import import_module

        InstrumentClass = getattr(import_module(f"qibolab.instruments.{lib}"), i_class)
        self.fpga = InstrumentClass(name, address, self.settings)

    def reload_settings(self):
        raise NotImplementedError

    def run_calibration(self, show_plots=False):  # pragma: no cover
        raise NotImplementedError

    def connect(self):
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def disconnect(self):
        raise NotImplementedError

    def execute_pulse_sequence(self, sequence, nshots=None):
        self.fpga.setup()
        avgi, avgq = self.fpga.play_sequence_and_acquire(sequence)
        return avgi, avgq
