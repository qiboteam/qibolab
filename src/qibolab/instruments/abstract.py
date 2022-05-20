from abc import ABC, abstractmethod
from qibolab.paths import qibolab_folder


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
        self.data_folder = qibolab_folder / "instruments" / "data"
        self.data_folder.mkdir(parents=True, exist_ok=True)

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
        header = f"InstrumentException with {instrument.signature}"
        full_msg = header + ": " + message
        super().__init__(full_msg)
