from qibo import K
from qibo.abstractions.states import AbstractState
from qibo.config import raise_error


class HardwareState(AbstractState):

    def __init__(self, nqubits=None):
        if nqubits > 1:
            raise_error(NotImplementedError, "Hardware device has one qubit.")
        super().__init__(nqubits)
        self.readout = None
        self.normalized_voltage = None
        self.min_voltage = None
        self.max_voltage = None

    @property
    def shape(self): # pragma: no cover
        raise_error(NotImplementedError)

    @property
    def dtype(self): # pragma: no cover
        raise_error(NotImplementedError)

    def symbolic(self, decimals=5, cutoff=1e-10, max_terms=20):  # pragma: no cover
        raise_error(NotImplementedError)

    def __array__(self): # pragma: no cover
        raise_error(NotImplementedError)

    def numpy(self): # pragma: no cover
        raise_error(NotImplementedError)

    def state(self, numpy=False, decimals=-1, cutoff=1e-10, max_terms=20):
        raise_error(NotImplementedError)

    @classmethod
    def from_readout(cls, readout, min_voltage, max_voltage):
        self.readout = readout
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage
        norm = max_voltage - min_voltage
        self.normalized_voltage = (readout[0] * 1e6 - min_voltage) / norm

    @classmethod
    def zero_state(cls, nqubits): # pragma: no cover
        raise_error(NotImplementedError)

    @classmethod
    def plus_state(cls, nqubits): # pragma: no cover
        raise_error(NotImplementedError)

    def copy(self, min_voltage=None, max_voltage=None):
        new = super().copy()
        new.readout = self.readout
        if min_voltage is not None:
            self.min_voltage = min_voltage
        if max_voltage is not None:
            self.max_voltage = max_voltage
        norm = self.max_voltage - self.min_voltage
        new.normalized_voltage = (self.readout[0] * 1e6 - self.min_voltage) / norm
        return new

    def to_density_matrix(self): # pragma: no cover
        raise_error(NotImplementedError)

    def probabilities(self, qubits=None, measurement_gate=None):
        p = self.normalized_voltage
        return K.cast([p, 1 - p], dtype="DTYPE")

    def measure(self, gate, nshots, registers=None): # pragma: no cover
        raise_error(NotImplementedError)

    def set_measurements(self, qubits, samples, registers=None): # pragma: no cover
        raise_error(NotImplementedError)

    def samples(self, binary=True, registers=False): # pragma: no cover
        raise_error(NotImplementedError)

    def frequencies(self, binary=True, registers=False): # pragma: no cover
        raise_error(NotImplementedError)

    def apply_bitflips(self, p0, p1=None): # pragma: no cover
        raise_error(NotImplementedError, "Noise simulation is not required for hardware.")

    def expectation(self, hamiltonian, normalize=False): # pragma: no cover
        raise_error(NotImplementedError)
