from abc import ABC, abstractmethod

class AbstractInstrument(ABC):
    """
    Parent class for all the instruments connected via TCPIP.
    
    Args:
        name (str): Instrument name.
        ip (str): IP network address.     
    """
    def __init__(self, name, ip):
        self.name = name
        self.ip = ip
        self.is_connected = False
        self.signature = f"{type(self).__name__}@{ip}"
        self.device = None

    @abstractmethod
    def connect(self):
        raise NotImplementedError

    @abstractmethod
    def setup(self):
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




class InstrumentException(Exception):
    def __init__(self, instrument: AbstractInstrument, message: str):
        self.instrument = instrument
        header = f"InstrumentException with {self.instrument.signature}"
        full_msg = header + ": " + message
        super().__init__(full_msg)
        self.instrument = instrument
