from abc import ABC, abstractmethod
from qibo.config import raise_error
from typing import Any


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

# With reference from qcodes.instruments.parameter
class BaseParameter:
    def __init__(self, name, default=None, vals=None, val_mapping=None, get_wrapper=None, validator=None):
        self.name = name
        self.vals = vals
        self.val_mapping = val_mapping
        self.value = default
        self.get_wrapper = get_wrapper
        self.validator = validator
    
    def get(self):
        if self.get_wrapper is None:
            return self.value
        else:
            return self.get_wrapper(self.value)

    def set(self, value):
        if self.validator is not None:
            if not self.validator(value, self.vals):
                raise RuntimeError("Invalid Value", value)
                
        self.value = value

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if len(args) == 0:
            return self.get()

        else:
            self.set(*args, **kwds)
            return None


def BoundsValidator(value, bounds):
    lbound, ubound = bounds
    return value >= lbound and value <= ubound

def EnumValidator(value, enum):
    return value in enum

class ParameterList(dict):

    def add_parameter(self, name: str, default=None, vals=None, val_mapping=None, get_wrapper=None, validator=None):
        self[name] = BaseParameter(name, default, vals, val_mapping, get_wrapper, validator)

    def __getattr__(self, key: str) -> Any:
        return self[key]
