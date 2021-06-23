from abc import ABC, abstractmethod
from qibo.config import raise_error
from typing import Any, Union


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


class GateSet(dict):

    from qibo import gates
    two_qubit_gates = [gates.CNOT]


    def set(self, gate, pulse_sequence: list):
        gate_name = gate.name

        if self.is_two_qubit_gate(gate):
            gate_name += "_{}".format(gate.control_qubits)

        self[gate_name] = pulse_sequence

    def set_from_dict(self, obj):
        for key, value in obj.items():
            self[key] = value
        
    def _is_two_qubit_gate(self, gate):
        return any(isinstance(gate, x) for x in self.two_qubit_gates)

    def get(self, gate):
        gate_name = gate.name

        if self.is_two_qubit_gate(gate):
            gate_name += "_{}".format(gate.control_qubits)

        return self[gate_name]


class Qubit(ParameterList):
    def __init__(self, id: int = None, qubit_frequency: float = 0, qubit_frequency_bounds: tuple = (0, 0), connected_qubits: list = [],
                 drive_channel: Union[tuple, int] = None, resonator_frequency: float = 0, resonator_frequency_bounds: list = (0, 0),
                 flux_channel: int = None, readout_channel: Union[tuple, int] = None, initial_gates: dict = {},
                 zero_iq_reference: tuple = (), one_iq_reference: tuple = ()):

        super().__init__()
        # static
        self["id"] = id
        self["qubit_frequency"] = qubit_frequency
        self["connected_qubits"] = connected_qubits
        self["drive_channel"] = drive_channel
        self["readout_channel"] = readout_channel

        # Optional for flux tunable qubits
        self["flux_channel"] = flux_channel
        # WIP: flux tuning term

        # configurables
        self.add_parameter("qubit_frequency", qubit_frequency, qubit_frequency_bounds, validator=BoundsValidator)
        self.add_parameter("resonator_frequency", resonator_frequency, resonator_frequency_bounds, validator=BoundsValidator)
        self.add_parameter("zero_iq_reference", zero_iq_reference)
        self.add_parameter("one_iq_reference", one_iq_reference)

        self["gates"] = GateSet()
        self.gates.set_from_dict(initial_gates)

    def edges(self):
        return [(self.id, q) for q in self.connected_qubits]
