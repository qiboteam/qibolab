from dataclasses import dataclass, field

from qibolab.instruments.abstract import AbstractInstrument


@dataclass
class BasicInstrumentDesign:
    """Instrument design that uses a single controller.

    Attributes:
        controller (:class:`qibolab.instruments.abstract.AbstractInstrument`): Instrument used for sending pulses and retrieving feedback.
        _is_connected (bool): Boolean that shows whether instruments are connected.
    """

    controller: AbstractInstrument
    channels: dict = field(default_factory=dict)
    _is_connected: bool = field(default=False, init=False)

    def connect(self):
        """Connect to all instruments."""
        if not self._is_connected:
            self.controller.connect()
        self._is_connected = True

    def setup(self, qubits, *args, **kwargs):
        """Load settings to instruments."""
        self.controller.setup(qubits, *args, **kwargs)

    def start(self):
        """Start all instruments."""
        self.controller.start()

    def stop(self):
        """Stop all instruments."""
        self.controller.stop()

    def disconnect(self):
        """Disconnect all instruments."""
        self.controller.disconnect()
        self._is_connected = False

    def play(self, *args, **kwargs):
        """Play a pulse sequence and retrieve feedback."""
        return self.controller.play(*args, **kwargs)

    def sweep(self, *args, **kwargs):
        """Play a pulse sequence while sweeping one or more parameters."""
        return self.controller.sweep(*args, **kwargs)
