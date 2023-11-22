from dataclasses import dataclass
from typing import Optional

from qibo.config import log

from qibolab.instruments.abstract import (
    Instrument,
    InstrumentException,
    InstrumentSettings,
)

RECONNECTION_ATTEMPTS = 3
"""Number of times to attempt connecting to instrument in case of failure."""


class DummyDevice:
    """Dummy device that does nothing but follows the QCoDeS interface."""

    ref_osc_source = None

    def set(self, name, value):
        """Set device property."""

    def get(self, name):
        """Get device property."""
        return 0

    def on(self):
        """Turn device on."""

    def off(self):
        """Turn device on."""

    def close(self):
        """Close connection with device."""


@dataclass
class LocalOscillatorSettings(InstrumentSettings):
    power: Optional[float] = None
    frequency: Optional[float] = None


class LocalOscillator(Instrument):
    """Abstraction for local oscillator instruments.

    Local oscillators are used to upconvert signals, when
    the controllers cannot send sufficiently high frequencies
    to address the qubits and resonators.
    They cannot be used to play or sweep pulses.
    """

    def __init__(self, name, address, reference_clock_source=None):
        super().__init__(name, address)
        self.device = None
        self.settings = LocalOscillatorSettings()
        # TODO: Maybe create an Enum for the reference clock
        self._reference_clock_source = reference_clock_source

    @property
    def frequency(self):
        return self.settings.frequency

    @frequency.setter
    def frequency(self, x):
        """Set frequency of the local oscillator.

        The value is cached in the :class:`qibolab.instruments.oscillator.LocalOscillatorSettings`
        dataclass. If we are connected to the instrument when the setter is called, it is also
        automatically uploaded to the instruments. If we are not connected the cached value
        is automatically uploaded when we connect.

        If the new value is the same with the cached value, it is not updated.
        """
        if self.frequency != x:
            self.settings.frequency = x
            if self.is_connected:
                self.device.set("frequency", x)

    @property
    def power(self):
        return self.settings.power

    @power.setter
    def power(self, x):
        """Set power of the local oscillator.

        The value is cached in the :class:`qibolab.instruments.oscillator.LocalOscillatorSettings`
        dataclass. If we are connected to the instrument when the setter is called, it is also
        automatically uploaded to the instruments. If we are not connected the cached value
        is automatically uploaded when we connect.

        If the new value is the same with the cached value, it is not updated.
        """
        if self.power != x:
            self.settings.power = x
            if self.is_connected:
                self.device.set("power", x)

    @property
    def reference_clock_source(self):
        return self._reference_clock_source

    @reference_clock_source.setter
    def reference_clock_source(self, x):
        """Switch the reference clock source of the local oscillator.

        The value is cached in the :class:`qibolab.instruments.oscillator.LocalOscillator`
        class. If we are connected to the instrument when the setter is called, it is also
        automatically uploaded to the instruments. If we are not connected the cached value
        is automatically uploaded when we connect.
        """
        self._reference_clock_source = x
        if self.is_connected:
            self.device.set("ref_osc_source", x)

    def create(self):
        """Create instance of physical device."""
        return DummyDevice()

    def connect(self):
        """Connects to the instrument using the IP address set in the runcard."""
        if not self.is_connected:
            for attempt in range(RECONNECTION_ATTEMPTS):
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
        else:
            self.settings.frequency = self.device.get("frequency")

        if self.settings.power is not None:
            self.device.set("power", self.settings.power)
        else:
            self.settings.power = self.device.get("power")

        if self.reference_clock_source is not None:
            self.reference_clock_source = self._reference_clock_source
        else:
            self._reference_clock_source = self.device.get("ref_osc_source")

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
