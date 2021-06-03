from abc import ABC, abstractmethod
from qibo.config import raise_error


class AbstractExperiment(ABC):

    def __init__(self):
        self.name = "abstract"
        self._connection = None
        self.static = None

    @property
    def connection(self):
        if self._connection is None:
            raise_error(RuntimeError, "Cannot establish connection.")
        return self._connection

    @abstractmethod
    def connect(self): # pragma: no cover
        raise_error(NotImplementedError)

    @abstractmethod
    def start(self): # pragma: no cover
        raise_error(NotImplementedError)

    @abstractmethod
    def stop(self): # pragma: no cover
        raise_error(NotImplementedError)

    @abstractmethod
    def upload(self): # pragma: no cover
        raise_error(NotImplementedError)

    @abstractmethod
    def download(self): # pragma: no cover
        raise_error(NotImplementedError)
