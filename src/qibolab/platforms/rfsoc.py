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
        super().__init__(name, runcard)
        self.name = name
        # self.resonator_freq = self.settings["instruments"][name]["settings"]["resonator_freq"]
        self.resonator_freq = self.settings["native_gates"]["single_qubit"][0]["MZ"]["frequency"]
        self.hardware_avg = self.settings["settings"]["hardware_avg"]

    def run_calibration(self, show_plots=False):  # pragma: no cover
        raise NotImplementedError

    def setup(self):
        config_setting = self.settings["instruments"][self.name]["settings"]
        config_gates = self.settings["native_gates"]["single_qubit"][0]
        cfg = config_setting | config_gates
        cfg["reps"] = self.settings["settings"]["hardware_avg"]
        self.instruments[self.name].setup(**cfg)

    def execute_pulse_sequence(self, sequence, nshots=None):
        config_setting = self.settings["instruments"][self.name]["settings"]
        config_gates = self.settings["native_gates"]["single_qubit"][0]
        config = config_setting | config_gates
        if nshots != None:
            config["reps"] = nshots
        else:
            config["reps"] = self.settings["settings"]["hardware_avg"]
        self.instruments[self.name].setup(**config)
        if nshots is None:
            nshots = self.hardware_avg

        i, q = self.instruments[self.name].play_sequence_and_acquire(sequence)
        msr = np.abs(i + 1j * q)
        phase = np.angle(i + q * 1j)
        return msr, phase, i, q
