"""RohdeSchwarz driver.

Supports the following Instruments:
    SGS100A

https://qcodes.github.io/Qcodes/api/generated/qcodes.instrument_drivers.rohde_schwarz.html#module-qcodes.instrument_drivers.rohde_schwarz.SGS100A
"""
import qcodes.instrument_drivers.rohde_schwarz.SGS100A as LO_SGS100A

from qibolab.instruments.oscillator import LocalOscillator


class SGS100A(LocalOscillator):
    def __init__(self, name, address, reference_clock_source="EXT"):
        super().__init__(name, address, reference_clock_source)

    def create(self):
        return LO_SGS100A.RohdeSchwarz_SGS100A(self.name, f"TCPIP0::{self.address}::5025::SOCKET")

    def __del__(self):
        self.disconnect()
