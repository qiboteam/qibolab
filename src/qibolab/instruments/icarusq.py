import pyvisa as visa
import urllib3

from qibolab.instruments.abstract import Instrument, InstrumentException
from qibolab.instruments.oscillator import LocalOscillator


class MCAttenuator(Instrument):
    """Driver for the MiniCircuit RCDAT-8000-30 variable attenuator."""

    def connect(self):
        pass

    def play(self, *args):
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

    def setup(self, attenuation: float, **kwargs):
        """Assigns the attenuation level on the attenuator.

        Arguments:
            attenuation(float
            ): Attenuation setting in dB. Ranges from 0 to 35.
        """
        self.attenuation = attenuation

    def start(self):
        pass

    def stop(self):
        pass

    def disconnect(self):
        pass


class QuicSyn(LocalOscillator):
    """Driver for the National Instrument QuicSyn Lite local oscillator."""

    def connect(self):
        if not self.is_connected:
            rm = visa.ResourceManager()
            try:
                self.device = rm.open_resource(self.address)
            except Exception as exc:
                raise InstrumentException(self, str(exc))
            self.is_connected = True

    @property
    def frequency(self):
        return float(self.device.query("FREQ?")) / 1e3

    @frequency.setter
    def frequency(self, freq):
        self.device.write("FREQ {:f}Hz".format(freq))

    def setup(self, frequency: float, **kwargs):
        """Sets the frequency in Hz."""
        if self.is_connected:
            self.device.write("0601")
            self.frequency = frequency

    def start(self):
        """Starts the instrument."""
        self.device.write("0F01")

    def stop(self):
        """Stops the instrument."""
        self.device.write("0F00")

    def __del__(self):
        self.disconnect()

    def disconnect(self):
        if self.is_connected:
            self.stop()
            self.device.close()
            self.is_connected = False

    def play(self, *args):
        pass
