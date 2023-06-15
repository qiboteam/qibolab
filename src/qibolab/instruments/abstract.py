import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

INSTRUMENTS_DATA_FOLDER = Path.home() / ".qibolab" / "instruments" / "data"


class Instrument(ABC):
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
        instruments_data_folder = INSTRUMENTS_DATA_FOLDER
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


class Controller(Instrument):
    """Instrument that can play pulses (using waveform generator)."""

    @abstractmethod
    def play(self, *args, **kwargs):
        """If the sequence is a PulseSequence play a pulse sequence and retrieve feedback.
        If the sequence is a List play a pulses sequences by unrolling and retrieve feedback

        Returns:
            If the sequence is a PulseSequence:
            (Dict[ResultType]) mapping the serial of the readout pulses used to
            the acquired :class:`qibolab.result.ExecutionResults` object.

            If the sequence is a List:
            (Dict[List[ResultType]) mapping the serial of the readout pulses used to a list of
            the acquired :class:`qibolab.result.ExecutionResults` object.
        """

    @abstractmethod
    def sweep(self, *args, **kwargs):
        """Play a pulse sequence while sweeping one or more parameters.

        Returns:
            (dict) mapping the serial of the readout pulses used to
            the acquired :class:`qibolab.result.ExecutionResults` object.
        """


class InstrumentException(Exception):
    def __init__(self, instrument: Instrument, message: str):
        header = f"InstrumentException with {instrument.signature}"
        full_msg = header + ": " + message
        super().__init__(full_msg)
