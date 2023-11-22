from dataclasses import dataclass, fields
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
    """Local oscillator parameters that are saved in the platform runcard."""

    power: Optional[float] = None
    frequency: Optional[float] = None
    ref_osc_source: Optional[str] = None

    @staticmethod
    def dict_factory(x):
        exclude_fields = ("ref_osc_source",)
        return {k: v for (k, v) in x if ((v is not None) and (k not in exclude_fields))}


def upload(func):
    """Decorator for parameter setters.

    The value of each parameter is cached in the :class:`qibolab.instruments.oscillator.LocalOscillator`
    object. If we are connected to the instrument when the setter is called, the new value is also
    automatically uploaded to the instruments. If we are not connected, the new value is cached
    and it is automatically uploaded after we connect.

    If the new value is the same with the cached value, it is not updated.
    """
    parameter = func.__name__

    def setter(self, x):
        if getattr(self, parameter) != x:
            setattr(self.settings, parameter, x)
            if self.is_connected:
                self.device.set(parameter, x)

    return setter


class LocalOscillator(Instrument):
    """Abstraction for local oscillator instruments.

    Local oscillators are used to upconvert signals, when
    the controllers cannot send sufficiently high frequencies
    to address the qubits and resonators.
    They cannot be used to play or sweep pulses.
    """

    def __init__(self, name, address, ref_osc_source=None):
        super().__init__(name, address)
        self.device = None
        self.settings = LocalOscillatorSettings(ref_osc_source=ref_osc_source)

    @property
    def frequency(self):
        return self.settings.frequency

    @frequency.setter
    @upload
    def frequency(self, x):
        """Set frequency of the local oscillator."""

    @property
    def power(self):
        return self.settings.power

    @power.setter
    @upload
    def power(self, x):
        """Set power of the local oscillator."""

    @property
    def ref_osc_source(self):
        return self.settings.ref_osc_source

    @ref_osc_source.setter
    @upload
    def ref_osc_source(self, x):
        """Switch the reference clock source of the local oscillator."""

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

        for fld in fields(self.settings):
            self.sync(fld.name)

    def sync(self, parameter):
        """Sync parameter value between our cache and the instrument.

        If the parameter value exists in our cache, it is uploaded to the instrument.
        If the value does not exist in our cache, it is downloaded

        Args:
            parameter (str): Parameter name to be synced.
        """
        value = getattr(self, parameter)
        if value is None:
            setattr(self.settings, parameter, value)
        else:
            self.device.set(parameter, value)

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
