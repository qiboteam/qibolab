from abc import ABC, abstractmethod

class Instrument(ABC):
    """
    Parent class for all the instruments
    """
    def __init__(self, ip):
        self._connected = False
        self.ip = ip
        self._signature = f"{type(self).__name__}@{ip}"
        self._driver = None
        self._connected = False
    
    @abstractmethod
    def connect(self):
        """
        Establish connection with the instrument.
        Initialize self._driver variable
        """
        raise NotImplementedError

    @property
    def signature(self):
        return self._signature

    @abstractmethod
    def close(self):
        """
        Close connection with the instrument.
        Set instrument values to idle values if required.
        """
        raise NotImplementedError


class InstrumentException(Exception):
    def __init__(self, instrument: Instrument, message: str):
        self.instrument = instrument
        header = f"InstrumentException with {self.instrument.signature}"
        full_msg = header + ": " + message
        super().__init__(full_msg)
        self.instrument = instrument