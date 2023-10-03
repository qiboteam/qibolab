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

    def connect(self):
        self.is_connected = True

    def setup(self, power=None, frequency=None, **kwargs):
        if power is not None:
            self.power = power
        if frequency is not None:
            self.frequency = frequency

    def start(self):
        pass

    def stop(self):
        pass

    def disconnect(self):
        self.is_connected = False
