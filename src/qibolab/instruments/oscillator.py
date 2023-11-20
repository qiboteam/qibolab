from dataclasses import dataclass
from typing import Optional

from qibo.config import log

from qibolab.instruments.abstract import (
    Instrument,
    InstrumentException,
    InstrumentSettings,
)


@dataclass
class LocalOscillatorSettings(InstrumentSettings):
    power: Optional[float] = None
    frequency: Optional[float] = None


class DummyDevice:
    ref_osc_source = None

    def set(self, name, value):
        """Set device property."""

    def on(self):
        """Turn device on."""

    def off(self):
        """Turn device on."""

    def close(self):
        """Close connection with device."""


class LocalOscillator(Instrument):
    """Abstraction for local oscillator instruments.

    Local oscillators are used to upconvert signals, when
    the controllers cannot send sufficiently high frequencies
    to address the qubits and resonators.
    They cannot be used to play or sweep pulses.
    """

    def __init__(self, name, address, reference_clock_source="EXT"):
        super().__init__(name, address)
        self.device = None
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
        self._reference_clock_source = x
        if self.is_connected:
            self.device.ref_osc_source = x

    def create(self):
        """Create instance of physical device."""
        return DummyDevice()

    def connect(self):
        """Connects to the instrument using the IP address set in the runcard."""
        if not self.is_connected:
            for attempt in range(3):
                try:
                    self.device = self.create()
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
            raise InstrumentException(self, "There is an open connection to the instrument already")
        self.upload()

    def upload(self):
        """Uploads cached setting values to the instruments."""
        if not self.is_connected:
            raise InstrumentException(self, "Cannot upload settings if instrument is not connected.")

        if self.settings.frequency is not None:
            self.device.set("frequency", self.settings.frequency)
        if self.settings.power is not None:
            self.device.set("power", self.settings.power)
        self.ref_osc_source = self._reference_clock_source

    def setup(self, **kwargs):
        """Update instrument settings.

        If the instrument is connected the value is automatically uploaded to the instrument.
        Otherwise the value is cached and will be uploaded when connection is established.

        Args:
            **kwargs: Instrument settings loaded from the runcard.
        """
        type_ = self.__class__
        properties = {p for p in dir(type_) if isinstance(getattr(type_, p), property)}
        for name, value in kwargs.items():
            if name not in properties:
                raise KeyError(f"Cannot set {name} to instrument {self.name} of type {type_.__name__}")
            setattr(self, name, value)

    def start(self):
        self.device.on()

    def stop(self):
        self.device.off()

    def disconnect(self):
        if self.is_connected:
            self.device.close()
            self.is_connected = False
