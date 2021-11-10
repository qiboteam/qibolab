from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error


class QibolabBackend(NumpyBackend): # pragma: no cover
    # hardware backend is not tested until `qiboicarusq` is available

    description = "" # TODO: Write proper description

    def __init__(self):
        super().__init__()
        self.name = "qiboicarusq"
        self.custom_gates = True
        import qibolab # pylint: disable=E0401
        self.is_hardware = True
        self.hardware_module = qibolab
        self.hardware_gates = qibolab.gates
        self.hardware_circuit = qibolab.circuit.HardwareCircuit

    def circuit_class(self, accelerators=None, density_matrix=False):
        if accelerators is not None:
            raise_error(NotImplementedError, "Hardware backend does not support "
                                             "multi-GPU configuration.")
        if density_matrix:
            raise_error(NotImplementedError, "Hardware backend does not support "
                                             "density matrix simulation.")
        from qibolab.circuit import HardwareCircuit
        return HardwareCircuit

    def create_einsum_cache(self, qubits, nqubits, ncontrol=None): # pragma: no cover
        raise_error(NotImplementedError, "`create_einsum_cache` method is "
                                         "not required for hardware backends.")

    def einsum_call(self, cache, state, matrix): # pragma: no cover
        raise_error(NotImplementedError, "`einsum_call` method is not required "
                                         "for hardware backends.")
