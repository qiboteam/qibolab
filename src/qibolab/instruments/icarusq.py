import urllib3
from qcodes_contrib_drivers.drivers.Valon.Valon_5015 import Valon5015 as Valon5015_LO

from qibolab.instruments.abstract import Instrument
from qibolab.instruments.oscillator import LocalOscillator


class MCAttenuator(Instrument):
    """Driver for the MiniCircuit RCDAT-8000-30 variable attenuator."""

    def connect(self):
        pass

    @property
    def attenuation(self):
        http = urllib3.PoolManager()
        res = http.request("GET", f"http://{self.address}/GETATT?")
        return float(res._body)

    @attenuation.setter
    def attenuation(self, attenuation: float):
        http = urllib3.PoolManager()
        http.request("GET", f"http://{self.address}/SETATT={attenuation}")

    def setup(self):
        pass

    def disconnect(self):
        pass


class Valon5015(LocalOscillator):
    """Driver for the Valon 5015 local oscillator."""

    def create(self):
        device = Valon5015_LO(
            self.name, address=f"TCPIP0::{self.address}::23::SOCKET", visalib="@py"
        )
        # The native qcodes driver does not have on/off functions required for the device class.
        # Here we add some lambda functions that corresponds to on/off.
        device.on = lambda: device.buffer_amplifiers_enabled(True)
        device.off = lambda: device.buffer_amplifiers_enabled(False)
        return device

    def __del__(self):
        self.disconnect()
