import qcodes.instrument_drivers.rohde_schwarz.SGS100A as LO_SGS100A

from qibolab._core.instruments.oscillator import LocalOscillator

__all__ = ["SGS100A"]


class SGS100A(LocalOscillator):
    """Driver to control the Rohde-Schwarz SGS100A local oscillator.

    This driver is using:
    https://qcodes.github.io/Qcodes/api/generated/qcodes.instrument_drivers.rohde_schwarz.html#module-qcodes.instrument_drivers.rohde_schwarz.SGS100A
    """

    def create(self):
        name = f"{type(self).__name__}{id(self)}"
        return LO_SGS100A.RohdeSchwarz_SGS100A(
            name, f"TCPIP0::{self.address}::5025::SOCKET"
        )
