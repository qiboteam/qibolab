
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
        self.resonator_freq = self.settings["instruments"][name]["settings"]["resonator_freq"]

    def run_calibration(self, show_plots=False):  # pragma: no cover
        raise NotImplementedError


    def execute_pulse_sequence(self, sequence, nshots=None):
        self.instruments[self.name].setup
        if nshots is None:
            nshots = self.hardware_avg
        i, q = self.instruments[self.name].play_sequence_and_acquire(sequence)
        msr=np.abs(i+1j*q)
        phase = np.angle(i+q*1j)
        return msr, phase, i, q
