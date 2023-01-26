from abc import ABC, abstractmethod


class AbstractInstrumentDesign(ABC):
    def __init__(self):
        self.is_connected = False

    @abstractmethod
    def connect(self):
        """Connect to all instruments."""

    @abstractmethod
    def setup(self, qubits):
        """Setup parameters of all instruments.

        Args:
            qubits (list): List of :class:`qibolab.platforms.utils.Qubit` objects
                that the instruments act on.
        """

    @abstractmethod
    def start(self):
        """Start all instruments."""

    @abstractmethod
    def stop(self):
        """Stop all instruments."""

    @abstractmethod
    def disconnect(self):
        """Disconnect all instruments."""

    @abstractmethod
    def play(self, *args, **kwargs):
        """Play an arbitrary pulse sequence and retrieve feedback."""

    @abstractmethod
    def sweep(self, *args, **kwargs):
        """Play an arbitrary pulse sequence while sweeping one or more parameters."""
