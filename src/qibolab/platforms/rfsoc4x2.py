from qibo.config import raise_error

from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibolab.instruments.tii import tii_rfsoc4x2

class RFSoCPlatform(AbstractPlatform):
    def run_calibration(self):
        raise_error(NotImplementedError)

    def execute_pulse_sequence(self, sequence: PulseSequence, nshots=None):
        tii_rfsoc4x2.setup()
        avgi, avgq = tii_rfsoc4x2.play_sequence_and_acquire()
        return avgi, avgq