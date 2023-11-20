"""RohdeSchwarz driver.

Supports the following Instruments:
    SGS100A

https://qcodes.github.io/Qcodes/api/generated/qcodes.instrument_drivers.rohde_schwarz.html#module-qcodes.instrument_drivers.rohde_schwarz.SGS100A
"""
import qcodes.instrument_drivers.rohde_schwarz.SGS100A as LO_SGS100A
from qibo.config import log

from qibolab.instruments.abstract import InstrumentException
from qibolab.instruments.oscillator import LocalOscillator, LocalOscillatorSettings


class SGS100A(LocalOscillator):
    def __init__(self, name, address, reference_clock_source="EXT"):
        super().__init__(name, address)
        self.device: LO_SGS100A = None
        self.settings = LocalOscillatorSettings()
        self._reference_clock_source = reference_clock_source

    @property
    def frequency(self):
        return self.settings.frequency

    @frequency.setter
    def frequency(self, x):
        if self.frequency != x:
            self.settings.frequency = x
            if self.is_connected:
                self.device.set("frequency", x)

    @property
    def power(self):
        return self.settings.power

    @power.setter
    def power(self, x):
        if self.power != x:
            self.settings.power = x
            if self.is_connected:
                self.device.set("power", x)

    @property
    def reference_clock_source(self):
        return self._reference_clock_source

    @reference_clock_source.setter
    def reference_clock_source(self, x):
        if self.reference_clock_source != x:
            self._reference_clock_source = x
            if self.is_connected:
                self.device.ref_osc_source = x

    def upload(self):
        """Uploads cached setting values to the instruments."""
        if not self.is_connected:
            raise RuntimeError("Cannot upload settings if instrument is not connected.")

        if self.settings.frequency is not None:
            self.device.set("frequency", self.settings.frequency)
        if self.settings.power is not None:
            self.device.set("power", self.settings.power)
        self.device.ref_osc_source = self._reference_clock_source

    def connect(self):
        """Connects to the instrument using the IP address set in the runcard."""
        if not self.is_connected:
            for attempt in range(3):
                try:
                    self.device = LO_SGS100A.RohdeSchwarz_SGS100A(self.name, f"TCPIP0::{self.address}::5025::SOCKET")
                    self.is_connected = True
                    break
                except KeyError as exc:
                    log.info(f"Unable to connect:\n{str(exc)}\nRetrying...")
                    self.name += "_" + str(attempt)
                except ConnectionError as exc:
                    log.info(f"Unable to connect:\n{str(exc)}\nRetrying...")
            if not self.is_connected:
                raise InstrumentException(self, f"Unable to connect to {self.name}")
        else:
            raise RuntimeError("There is an open connection to the instrument already")
        self.upload()

    def start(self):
        self.device.on()

    def stop(self):
        self.device.off()

    def disconnect(self):
        if self.is_connected:
            self.device.close()
            self.is_connected = False

    def __del__(self):
        self.disconnect()
