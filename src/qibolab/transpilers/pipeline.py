from dataclasses import dataclass
from typing import List

from qibo.config import log

from qibolab.transpilers.abstract import Transpiler
from qibolab.transpilers.fusion import Fusion, Rearrange
from qibolab.transpilers.gate_decompositions import NativeGates
from qibolab.transpilers.star_connectivity import StarConnectivity


@dataclass
class Pipeline(Transpiler):
    """Transpiler consisting of a list of smaller transpilers that are applied sequentially."""

    transpilers: List[Transpiler]
    verbose: bool = False

    def is_satisfied(self, circuit):
        for transpiler in self.transpilers:
            if not transpiler.is_satisfied(circuit):
                return False
        return True

    def __call__(self, circuit):
        if self.verbose:
            log.info("Transpiling circuit.")
        nqubits = circuit.nqubits
        total_map = list(range(nqubits))
        for transpiler in self.transpilers:
            circuit, qubit_map = transpiler(circuit)
            total_map = [qubit_map[total_map[i]] for i in range(nqubits)]
        return circuit, total_map

    @classmethod
    def default(cls, two_qubit_natives, middle_qubit=2, fuse_one_qubit=False, verbose=False):
        """Default transpiler used by :class:`qibolab.backends.QibolabBackend`."""
        transpilers = [Rearrange(max_qubits=2)]
        # add SWAPs to satisfy connectivity constraints
        transpilers.append(StarConnectivity(middle_qubit))
        # two-qubit gates to native
        transpilers.append(NativeGates(two_qubit_natives, translate_single_qubit=False, verbose=verbose))
        # Optional: fuse one-qubit gates to reduce circuit depth
        if fuse_one_qubit:
            transpilers.append(Fusion(max_qubits=1))
        # one-qubit gates to native
        transpilers.append(NativeGates(two_qubit_natives, translate_single_qubit=True, verbose=verbose))
        return cls(transpilers, verbose)
