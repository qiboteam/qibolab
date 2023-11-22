from dataclasses import dataclass
from enum import Enum
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
    """Local oscillator parameters that are saved in the platform runcard."""

    power: Optional[float] = None
    frequency: Optional[float] = None


class LocalOscillatorParameters(Enum):
    """Local oscillator parameters.

    The value corresponding to each parameter is ``True`` if the parameter
    is in :class:`qibolab.instruments.oscillator.LocalOscillatorSettings`
    and ``False`` otherwise, if the parameter is only a property of the instrument.
    """

    power = True
    frequency = True
    ref_osc_source = False


def upload(func):
    """Decorator for parameter setters."""
    parameter = func.__name__
    is_setting = getattr(LocalOscillatorParameters, parameter).value

    def setter(self, x):
        if getattr(self, parameter) != x:
            if is_setting:
                setattr(self.settings, parameter, x)
            else:
                setattr(self, f"_{parameter}", x)
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
        self.settings = LocalOscillatorSettings()
        # TODO: Maybe create an Enum for the reference clock
        self._ref_osc_source = ref_osc_source

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
    def ref_osc_source(self):
        return self._ref_osc_source

    @ref_osc_source.setter
    def ref_osc_source(self, x):
        """Switch the reference clock source of the local oscillator.

        The value is cached in the :class:`qibolab.instruments.oscillator.LocalOscillator`
        class. If we are connected to the instrument when the setter is called, it is also
        automatically uploaded to the instruments. If we are not connected the cached value
        is automatically uploaded when we connect.
        """
        if self.ref_osc_source != x:
            self._ref_osc_source = x
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

        for parameter in LocalOscillatorParameters:
            self.sync(parameter)

    def sync(self, parameter):
        """Sync parameter value between our cache and the instrument.

        If the parameter value exists in our cache, it is uploaded to the instrument.
        If the value does not exist in our cache, it is downloaded

        Args:
            parameter: ``LocalOscillatorParameter`` to be synced.
        """
        name, is_setting = parameter.name, parameter.value
        if is_setting:
            value = getattr(self.settings, name)
        else:
            value = getattr(self, name)

        if value is None:
            if is_setting:
                setattr(self.settings, name, self.device.get(name))
            else:
                setattr(self, f"_{name}", self.device.get(name))
        else:
            self.device.set(name, value)

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
