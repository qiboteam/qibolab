from dataclasses import dataclass

from qibo import gates

from qibolab.transpilers.abstract import AbstractTranspiler


@dataclass
class FusionTranspiler(AbstractTranspiler):
    max_qubits: int = 1

    def is_satisfied(self, circuit):
        return True

    def transpile(self, circuit):
        return circuit.fuse(max_qubits=self.max_qubits), list(range(circuit.nqubits))


@dataclass
class RearrangeTranspiler(FusionTranspiler):
    """Rearranges gates using qibo's fusion algorithm.

    May reduce number of SWAPs when fixing for connectivity.
    """

    def transpile(self, circuit):
        fcircuit = circuit.fuse(max_qubits=self.max_qubits)
        new = circuit.__class__(circuit.nqubits)
        for fgate in fcircuit.queue:
            if isinstance(fgate, gates.FusedGate):
                new.add(fgate.gates)
            else:
                new.add(fgate)
        return new, list(range(circuit.nqubits))
