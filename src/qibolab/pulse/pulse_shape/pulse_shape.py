from abc import ABC, abstractmethod
from qibo.config import raise_error

class PulseShape(ABC):
    """Abstract class for pulse shapes"""
    
    @property
    @abstractmethod
    def envelope_i(self): # pragma: no cover
        raise_error(NotImplementedError)

    @property
    @abstractmethod
    def envelope_q(self): # pragma: no cover
        raise_error(NotImplementedError)