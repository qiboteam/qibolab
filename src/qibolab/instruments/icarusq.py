import urllib3
from icarusq_rfsoc_driver.quicsyn import QuicSyn as LO_QuicSyn  # pylint: disable=E0401

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


class QuicSyn(LocalOscillator):
    """Driver for the National Instrument QuicSyn Lite local oscillator."""

    def create(self):
        return LO_QuicSyn(self.name, self.address)

    def __del__(self):
        self.disconnect()
