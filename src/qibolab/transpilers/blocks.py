from qibo import Circuit
from qibo.config import raise_error


class Block:
    """A block contains a subset of gates acting on two qubits.

    Args:
        qubits (tuple): qubits where the block is acting.
        gates (list): list of gates that compose the block.
        name (str): name of the block.
    """

    def __init__(self, qubits: tuple, gates: list, name: str = None):
        self.qubits = qubits
        self.gates = gates
        self.name = name

    def rename(self, name):
        self.name = name

    def add_gate(self, gate):
        self.gates.append(gate)

    def get_qubits(self):
        return self.qubits

    def get_gates(self):
        return self.gates

    def get_name(self):
        return self.name

    # TODO
    def kak_decompose(self):
        """Return KAK decomposition of the block"""
        raise_error(NotImplementedError)


def block_decomposition(circuit: Circuit):
    """Decompose a circuit into blocks of gates acting on two qubits.

    Args:
        circuit (qibo.Circuit): circuit to be decomposed.

    Return:
        blocks (list): list of blocks that act on two qubits.
    """
