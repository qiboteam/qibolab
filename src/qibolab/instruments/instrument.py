from abc import ABC, abstractmethod

class Instrument(ABC):
    """
    Parent class for all the instruments
    """
    def __init__(self):
        self._connected = False
        # Implement signature as InstrumentClassName@ip_address
        self._signature = None
        self._driver = None
    
    @abstractmethod
    def connect(self):
        """
        Establish connection with the instrument.
        Initialize self._driver variable
        """
        raise NotImplementedError

    @property
    def _signature(self):
        return self._signature

    @abstractmethod
    def close(self):
        """
        Close connection with the instrument.
        Set instrument values to idle values if required.
        """
        raise NotImplementedError