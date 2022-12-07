import numpy as np
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.config import raise_error

from qibolab.transpilers.unitary_decompositions import (
    two_qubit_decomposition,
    u3_decomposition,
)


class GateDecompositions:
    """Abstract data structure that holds decompositions of gates."""

    def __init__(self):
        self.decompositions = {}

    def add(self, gate, decomposition):
        """Register a decomposition for a gate."""
        self.decompositions[gate] = decomposition

    def count_2q(self, gate):
        """Count the number of two-qubit gates in the decomposition of the given gate."""
        decomposition = self.decompositions[gate.__class__]
        return len(tuple(g for g in decomposition if len(g.qubits) > 1))

    def count_1q(self, gate):
        """Count the number of single qubit gates in the decomposition of the given gate."""
        decomposition = self.decompositions[gate.__class__]
        return len(tuple(g for g in decomposition if len(g.qubits) == 1))

    def __call__(self, gate):
        """Decompose a gate."""
        decomposition = self.decompositions[gate.__class__]
        return [g.on_qubits({i: q for i, q in enumerate(gate.qubits)}) for g in decomposition]


# register the iSWAP decompositions
iswap_dec = GateDecompositions()
iswap_dec.add(
    gates.CNOT,
    [
        gates.H(0),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi, 0, np.pi),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi / 2, np.pi / 2, -np.pi),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi / 2),
    ],
)
iswap_dec.add(
    gates.CZ,
    [
        gates.H(0),
        gates.H(1),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi, 0, np.pi),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi / 2, np.pi / 2, -np.pi),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi / 2),
        gates.H(1),
    ],
)
iswap_dec.add(
    gates.SWAP,
    [
        gates.iSWAP(0, 1),
        gates.RX(1, np.pi / 2),
        gates.iSWAP(0, 1),
        gates.RX(0, np.pi / 2),
        gates.iSWAP(0, 1),
        gates.RX(1, np.pi / 2),
    ],
)
iswap_dec.add(gates.iSWAP, [gates.iSWAP(0, 1)])

# register CZ decompositions
cz_dec = GateDecompositions()
cz_dec.add(gates.CNOT, [gates.H(1), gates.CZ(0, 1), gates.H(1)])
cz_dec.add(gates.CZ, [gates.CZ(0, 1)])
cz_dec.add(
    gates.SWAP,
    [
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(1),
        gates.H(0),
        gates.CZ(1, 0),
        gates.H(0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(1),
    ],
)
cz_dec.add(
    gates.iSWAP,
    [
        gates.U3(0, np.pi / 2.0, 0, -np.pi / 2.0),
        gates.U3(1, np.pi / 2.0, 0, -np.pi / 2.0),
        gates.CZ(0, 1),
        gates.H(0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(0),
        gates.H(1),
    ],
)

# register other optimized gate decompositions
opt_dec = GateDecompositions()
opt_dec.add(
    gates.SWAP,
    [
        gates.H(0),
        gates.SDG(0),
        gates.SDG(1),
        gates.iSWAP(0, 1),
        gates.CZ(0, 1),
        gates.H(1),
    ],
)
