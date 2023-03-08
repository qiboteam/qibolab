import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from qibolab.paths import user_folder


class AbstractInstrument(ABC):
    """
    Parent class for all the instruments connected via TCPIP.

    Args:
        name (str): Instrument name.
        address (str): Instrument network address.
    """

    def __init__(self, name, address):
        self.name: str = name
        self.address: str = address
        self.is_connected: bool = False
        self.signature: str = f"{type(self).__name__}@{address}"
        # create local storage folder
        instruments_data_folder = user_folder / "instruments" / "data"
        instruments_data_folder.mkdir(parents=True, exist_ok=True)
        # create temporary directory
        self.tmp_folder = tempfile.TemporaryDirectory(dir=instruments_data_folder)
        self.data_folder = Path(self.tmp_folder.name)

    @abstractmethod
    def connect(self):
        raise NotImplementedError

    @abstractmethod
    def setup(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def start(self):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError

    @abstractmethod
    def disconnect(self):
        raise NotImplementedError

    def play(self, *args, **kwargs):
        """Play a pulse sequence and retrieve feedback.

        Returns:
            (dict) mapping the serial of the readout pulses used to
            the acquired :class:`qibolab.result.ExecutionResults` object.
        """
        raise NotImplementedError(f"Instrument {self.name} does not support play.")

    def sweep(self, *args, **kwargs):
        """Play a pulse sequence while sweeping one or more parameters.

        Returns:
            (dict) mapping the serial of the readout pulses used to
            the acquired :class:`qibolab.result.ExecutionResults` object.
        """
        raise NotImplementedError(f"Instrument {self.name} does not support sweep.")


class LocalOscillator(AbstractInstrument):
    """Abstraction for local oscillator instruments.

    Local oscillators are used to upconvert signals, when
    the controllers cannot send sufficiently high frequencies
    to address the qubits and resonators.
    """

    def play(self, *args, **kwargs):
        """Local oscillators do not play pulses."""

    def sweep(self, *args, **kwargs):
        """Local oscillators do not play pulses."""


class InstrumentException(Exception):
    def __init__(self, instrument: AbstractInstrument, message: str):
        header = f"InstrumentException with {instrument.signature}"
        full_msg = header + ": " + message
        super().__init__(full_msg)
