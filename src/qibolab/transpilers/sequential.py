from dataclasses import dataclass
from typing import List

from qibo.config import log

from qibolab.transpilers.abstract import AbstractTranspiler
from qibolab.transpilers.fusion import FusionTranspiler, RearrangeTranspiler
from qibolab.transpilers.gate_decompositions import NativeGateTranspiler
from qibolab.transpilers.star_connectivity import StarConnectivityTranspiler


@dataclass
class SequentialTranspiler(AbstractTranspiler):
    transpilers: List[AbstractTranspiler]
    verbose: bool = False

    def is_satisfied(self, circuit):
        for transpiler in transpilers:
            if not transpiler.is_satisfied(circuit):
                return False
        return True

    def transpile(self, circuit):
        if self.verbose:
            log.info("Transpiling circuit.")
        for transpiler in self.transpilers:
            circuit = transpiler.transpile(circuit)
        return circuit

    @classmethod
    def default(cls, two_qubit_natives, middle_qubit=2, fuse_one_qubit=False, verbose=False):
        transpilers = [RearrangeTranspiler(2)]
        # add SWAPs to satisfy connectivity constraints
        transpilers.append(StarConnectivityTranspiler(middle_qubit))
        # two-qubit gates to native
        transpilers.append(NativeGateTranspiler(two_qubit_natives, translate_single_qubit=False, verbose=verbose))
        # Optional: fuse one-qubit gates to reduce circuit depth
        if fuse_one_qubit:
            transpilers.append(FusionTranspiler(1))
        # one-qubit gates to native
        transpilers.append(NativeGateTranspiler(two_qubit_natives, translate_single_qubit=True, verbose=verbose))
        return cls(transpilers, verbose)
