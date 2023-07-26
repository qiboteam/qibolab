from qibolab.instruments.abstract import Instrument


class LocalOscillator(Instrument):
    """Abstraction for local oscillator instruments.

    Local oscillators are used to upconvert signals, when
    the controllers cannot send sufficiently high frequencies
    to address the qubits and resonators.
    They cannot be used to play or sweep pulses.
    """

    def __init__(self, name, address):
        super().__init__(name, address)
        self._power: float
        self._frequency: float

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, x):
        self._frequency = x

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, x):
        self._power = x

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
