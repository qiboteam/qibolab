from abc import ABC, abstractmethod
from qibo.config import raise_error


class Instrument(ABC):
    """
    Parent class for all the instruments connected via TCPIP.
    """

    def __init__(self, ip):
        self._connected = False
        self.ip = ip
        self._signature = f"{type(self).__name__}@{ip}"
        self.device = None
    
    @abstractmethod
    def connect(self):  # pragma: no cover
        """
        Establish connection with the instrument.
        Initialize self.device variable
        """
        raise_error(NotImplementedError)

    @property
    def signature(self):
        return self._signature

    @abstractmethod
    def close(self):  # pragma: no cover
        """
        Close connection with the instrument.
        Set instrument values to idle values if required.
        """
        raise_error(NotImplementedError)


class InstrumentException(Exception):  # pragma: no cover

    def __init__(self, instrument: Instrument, message: str):
        self.instrument = instrument
        header = f"InstrumentException with {self.instrument.signature}"
        full_msg = header + ": " + message
        super().__init__(full_msg)