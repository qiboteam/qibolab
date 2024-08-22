from abc import abstractmethod
from typing import Optional

from pydantic import Field
from qcodes.instrument import Instrument as QcodesInstrument

from qibolab.instruments.abstract import Instrument, InstrumentSettings

RECONNECTION_ATTEMPTS = 3
"""Number of times to attempt connecting to instrument in case of failure."""


class LocalOscillatorSettings(InstrumentSettings):
    """Local oscillator parameters that are saved in the platform runcard."""

    power: Optional[float] = None
    frequency: Optional[float] = None
    ref_osc_source: Optional[str] = None


def _setter(instrument, parameter, value):
    """Set value of a setting.

    The value of each parameter is cached in the :class:`qibolab.instruments.oscillator.LocalOscillator`.
    If we are connected to the instrument when the setter is called, the new value is also
    automatically uploaded to the instruments. If we are not connected, the new value is cached
    and it is automatically uploaded after we connect.
    If the new value is the same with the cached value, it is not updated.
    """
    if getattr(instrument, parameter) != value:
        setattr(instrument.settings, parameter, value)
        if instrument.is_connected:
            instrument.device.set(parameter, value)


def _property(parameter):
    """Create an instrument property."""
    return property(
        lambda self: getattr(self.settings, parameter),
        lambda self, value: _setter(self, parameter, value),
    )


class LocalOscillator(Instrument):
    """Abstraction for local oscillator instruments.

    Local oscillators are used to upconvert signals, when the
    controllers cannot send sufficiently high frequencies to address the
    qubits and resonators. They cannot be used to play or sweep pulses.
    """

    device: Optional[QcodesInstrument] = None
    settings: Optional[InstrumentSettings] = Field(
        default_factory=lambda: LocalOscillatorSettings()
    )

    frequency = _property("frequency")
    power = _property("power")
    ref_osc_source = _property("ref_osc_source")

    @abstractmethod
    def create(self) -> QcodesInstrument:
        """Create instance of physical device."""

    def connect(self):
        """Connect to the instrument."""
        if not self.is_connected:
            self.device = self.create()
            self.is_connected = True
            if not self.is_connected:
                raise RuntimeError(f"Unable to connect to {self.name}.")
        else:
            raise RuntimeError(
                f"There is an open connection to the instrument {self.name}."
            )

        assert self.settings is not None
        for fld in self.settings.model_fields:
            self.sync(fld)

        self.device.on()

    def disconnect(self):
        if self.is_connected:
            assert self.device is not None
            self.device.off()
            self.device.close()
            self.is_connected = False

    def sync(self, parameter):
        """Sync parameter value between our cache and the instrument.

        If the parameter value exists in our cache, it is uploaded to the instrument.
        If the value does not exist in our cache, it is downloaded

        Args:
            parameter (str): Parameter name to be synced.
        """
        value = getattr(self, parameter)
        assert self.device is not None
        if value is None:
            setattr(self.settings, parameter, self.device.get(parameter))
        else:
            self.device.set(parameter, value)

    def setup(self, **kwargs):
        """Update instrument settings.

        If the instrument is connected the value is automatically uploaded to the instrument.
        Otherwise the value is cached and will be uploaded when connection is established.

        Args:
            **kwargs: Instrument settings loaded from the runcard.
        """
        assert self.settings is not None
        for name, value in kwargs.items():
            if name not in self.settings.model_fields:
                raise KeyError(f"Cannot set {name} to instrument {self.name}")
            setattr(self, name, value)
