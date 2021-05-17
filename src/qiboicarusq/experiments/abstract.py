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
        return self.connection

    @abstractmethod
    def connect(self):
        raise_error(NotImplementedError)

    @abstractmethod
    def start(self):
        raise_error(NotImplementedError)

    @abstractmethod
    def stop(self):
        raise_error(NotImplementedError)

    @abstractmethod
    def upload(self):
        raise_error(NotImplementedError)

    @abstractmethod
    def download(self):
        raise_error(NotImplementedError)
