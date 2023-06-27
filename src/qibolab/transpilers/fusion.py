from qibo import gates

from qibolab.transpilers.abstract import Optimizer


class Fusion(Optimizer):
    """Apply gate fusion up to the given ``max_qubits``."""

    def __init__(self, max_qubits: int = 1):
        self.max_qubits = max_qubits

    def __call__(self, circuit):
        return circuit.fuse(max_qubits=self.max_qubits), list(range(circuit.nqubits))


class Rearrange(Optimizer):
    """Rearranges gates using qibo's fusion algorithm.

    May reduce number of SWAPs when fixing for connectivity
    but this has not been tested.
    """

    def __init__(self, max_qubits: int = 1):
        self.max_qubits = max_qubits

    def __call__(self, circuit):
        fcircuit = circuit.fuse(max_qubits=self.max_qubits)
        new = circuit.__class__(circuit.nqubits)
        for fgate in fcircuit.queue:
            if isinstance(fgate, gates.FusedGate):
                new.add(fgate.gates)
            else:
                new.add(fgate)
        return new
