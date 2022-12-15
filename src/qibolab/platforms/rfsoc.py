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
    """
    def __init__(self, name, runcard):
        log.info(f"Loading platform {name} from runcard {runcard}")
        self.name = name
        self.runcard = runcard
        self.is_connected = False
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
    """



    def run_calibration(self, show_plots=False):  # pragma: no cover
        raise NotImplementedError

    

    def execute_pulse_sequence(self, sequence, nshots=None):
        avgi, avgq = self.instruments[self.name].play_sequence_and_acquire(sequence)
        print("i y q: ", avgi, avgq )
        i = avgi[0]
        q = avgq[0]
        msr=np.abs(i+1j*q)
        phase = np.angle(i+q*1j)
        return msr, phase, i, q
