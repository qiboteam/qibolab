from abc import ABC, abstractmethod
from qibo.config import raise_error
from typing import Any, Union, List, Tuple


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
    def __init__(self, name: str, default=None, vals=None, val_mapping=None, get_wrapper=None, validator=None):
        """An adjustable parameter to be used in the experiment.

        Args:
        name: Parameter name
        default: Parameter default value
        vals: To be used to validate new values for the parameter
        val_mapping: Iterable to be used by get_wrapper when get is called
        get_wrapper: Function to format parameter value when used
        validator: Function that validates new parameter value against vals
        @see BoundsValidator, EnumValidator for examples
        """
        self.name = name
        self.vals = vals
        self.val_mapping = val_mapping
        self.value = default
        self.get_wrapper = get_wrapper
        self.validator = validator
    
    def get(self):
        """Fetches the parameter value.

        if get_wrapper is available, return the value returned by the get_wrapped function
        """
        if self.get_wrapper is None:
            return self.value
        else:
            return self.get_wrapper(self.value, self.val_mapping)

    def set(self, value):
        """Sets a new value for the parameter.

        if validator is set, first validate the new value and raise an error is value is invalid.
        """
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

class ParameterList(dict[str, BaseParameter]):
    """Dictionary class that holds BaseParameters.
    """
    def add_parameter(self, name: str, default=None, vals=None, val_mapping=None, get_wrapper=None, validator=None):
        """Adds a new parameter to the dictionary
        
        @see BaseParameter for arguments
        """
        self[name] = BaseParameter(name, default, vals, val_mapping, get_wrapper, validator)

    def __getattr__(self, key: str) -> Any:
        return self[key]


class GateSet(dict):
    """Dictionary containing the PulseSequence objects for each gate.
    """

    from qibo import gates
    two_qubit_gates = [gates.CNOT]


    def set(self, gate, pulse_sequence: list):
        """Sets a qibo gate into the register with its corresponding pulse sequence.
        If a two qubit gate is set, the id of the control qubit is also set for reference.

        Args:
        gate: Qibo gate object
        pulse_sequence: List of PulseSequence objects that represent the gate
        """
        gate_name = gate.name

        if self.is_two_qubit_gate(gate):
            gate_name += "_{}".format(gate.control_qubits)

        self[gate_name] = pulse_sequence

    def set_from_dict(self, obj: dict):
        for key, value in obj.items():
            self[key] = value
        
    def _is_two_qubit_gate(self, gate):
        return any(isinstance(gate, x) for x in self.two_qubit_gates)

    def get(self, gate) -> List[Any]:
        """Fetches the pulse representation of a qibo gate
        Note: For parameterized gates like RX, RY or RZ, this function does not set the parameter.

        Args:
        gate: Qibo gate object

        Returns:
        List of PulseSequence objects
        """
        gate_name = gate.name

        if self.is_two_qubit_gate(gate):
            gate_name += "_{}".format(gate.control_qubits)

        return self[gate_name]


class Qubit(ParameterList):
    """Qubit representation with parameters for control and readout
    """
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

    def edges(self) -> List[Tuple]:
        """Fetches a list of edge tuples representing qubit connectivity.
        """
        return [(self.id, q) for q in self.connected_qubits]
