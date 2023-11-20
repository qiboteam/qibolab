from dataclasses import dataclass
from typing import Optional

from qibolab.instruments.abstract import Instrument, InstrumentSettings


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

    def __init__(self, name, address):
        super().__init__(name, address)
        self.settings = LocalOscillatorSettings()

    @property
    def frequency(self):
        return self.settings.frequency

    @frequency.setter
    def frequency(self, x):
        self.settings.frequency = x

    @property
    def power(self):
        return self.settings.power

    @power.setter
    def power(self, x):
        self.settings.power = x

    def upload(self):
        """Uploads cached setting values to the instruments."""

    def connect(self):
        self.is_connected = True
        self.upload()

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
        pass

    def stop(self):
        pass

    def disconnect(self):
        self.is_connected = False
