import os
from qibolab.platform import Platform
from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error


class QibolabBackend(NumpyBackend): # pragma: no cover

    description = "" # TODO: Write proper description

    def __init__(self):
        super().__init__()
        self.name = "qibolab"
        self.custom_gates = True
        self.is_hardware = True
        self.set_platform(os.environ.get("QIBOLAB_PLATFORM", "tiiq"))

    def set_platform(self, platform):
        self.platform = Platform(platform)

    def get_platform(self):
        return self.platform.name

    def circuit_class(self, accelerators=None, density_matrix=False):
        if accelerators is not None:
            raise_error(NotImplementedError, "Hardware backend does not support "
                                             "multi-GPU configuration.")
        if density_matrix:
            raise_error(NotImplementedError, "Hardware backend does not support "
                                             "density matrix simulation.")
        from qibolab.circuit import HardwareCircuit
        return HardwareCircuit

    def create_gate(self, cls, *args, **kwargs):
        from qibolab import gates
        return getattr(gates, cls.__name__)(*args, **kwargs)

    def create_einsum_cache(self, qubits, nqubits, ncontrol=None): # pragma: no cover
        raise_error(NotImplementedError, "`create_einsum_cache` method is "
                                         "not required for hardware backends.")

    def einsum_call(self, cache, state, matrix): # pragma: no cover
        raise_error(NotImplementedError, "`einsum_call` method is not required "
                                         "for hardware backends.")
