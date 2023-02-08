from qibolab.instruments.abstract import LocalOscillator


class DummyLocalOscillator(LocalOscillator):
    """Dummy local oscillator driver.

    Useful for using with the Quantum Machines simulator.
    """

    def __init__(self, name, address):
        super().__init__(name, address)
        self.power: float
        self.frequency: float

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
