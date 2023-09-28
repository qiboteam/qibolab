from copy import copy, deepcopy

from qibo import Circuit
from qibo.config import raise_error
from qibo.gates import Gate


class BlockingError(Exception):
    """Raise when an error occurs in the blocking procedure"""


class Block:
    """A block contains a subset of gates acting on two qubits.

    Args:
        qubits (tuple): qubits where the block is acting.
        gates (list): list of gates that compose the block.
        name (str): name of the block.
        entangled (bool): true if there is at least a two qubit gate in the block.
    """

    def __init__(self, qubits: tuple, gates: list, name: str = None, entangled: bool = True):
        self._qubits = qubits
        self.gates = gates
        self.name = name
        self.entangled = entangled

    def rename(self, name):
        self.name = name

    def add_gate(self, gate: Gate):
        if not gate.qubits == self.qubits:
            raise BlockingError(
                "Gate acting on qubits {} can't be added to block acting on qubits {}.".format(
                    gate.qubits, self._qubits
                )
            )
        self.gates.append(gate)
        if len(gate.qubits) == 2:
            self.entangled = True

    def count_2q_gates(self):
        return count_2q_gates(self.gates)

    @property
    def qubits(self):
        return tuple(sorted(self._qubits))

    @qubits.setter
    def qubits(self, qubits):
        self._qubits = qubits

    def info(self):
        print("Block Name: ", self.name)
        print("Qubits: ", self.qubits)
        print("Gates: ", self.gates)
        print("Number of two qubits gates: ", self.count_2q_gates())
        print("Entangled: ", self.entangled)

    # TODO
    def kak_decompose(self):  # pragma: no cover
        """Return KAK decomposition of the block.
        This should be done only if the block is entangled and the number of
        two qubit gates is higher than the number after the decomposition.
        """
        raise_error(NotImplementedError)


def fuse_blocks(block_1: Block, block_2: Block, name=None):
    """Fuse two gate blocks, the qubits they are acting on must coincide.

    Args:
        block_1 (.transpilers.blocks.Block): first block.
        block_2 (.transpilers.blocks.Block): second block.
        name (str): name of the fused block.

    Return:
        fused_block (.transpilers.blocks.Block): fusion of the two input blocks.
    """
    if not block_1.qubits == block_2.qubits:
        raise BlockingError("In order to fuse two blocks their qubits must coincide.")
    entangled = block_1.entangled or block_2.entangled
    return Block(qubits=block_1.qubits, gates=block_1.gates + block_2.gates, name=name, entangled=entangled)


def commute(block_1: Block, block_2: Block):
    """Check if two blocks commute (share qubits).

    Args:
        block_1 (.transpilers.blocks.Block): first block.
        block_2 (.transpilers.blocks.Block): second block.

    Return:
        True if the two blocks don't share any qubit.
        False otherwise.
    """
    for qubit in block_1.qubits:
        if qubit in block_2.qubits:
            return False
    return True


def block_decomposition(circuit: Circuit, fuse: bool = True):
    """Decompose a circuit into blocks of gates acting on two qubits.

    Args:
        circuit (qibo.models.Circuit): circuit to be decomposed.
        fuse (bool): fuse adjacent blocks acting on the same qubits.

    Return:
        blocks (list): list of blocks that act on two qubits.
    """
    if circuit.nqubits < 2:
        raise BlockingError("Only circuits with at least two qubits can be decomposed with block_decomposition.")
    initial_blocks = initial_block_decomposition(circuit)
    if not fuse:
        return initial_blocks
    blocks = []
    while len(initial_blocks) > 0:
        first_block = initial_blocks[0]
        initial_blocks.remove(first_block)
        if len(initial_blocks) > 0:
            following_blocks = deepcopy(initial_blocks)
            for idx, second_block in enumerate(following_blocks):
                try:
                    first_block = fuse_blocks(first_block, second_block)
                    initial_blocks.remove(initial_blocks[idx])
                except BlockingError:
                    if not commute(first_block, second_block):
                        break
        blocks.append(first_block)
    return blocks


def initial_block_decomposition(circuit: Circuit):
    """Decompose a circuit into blocks of gates acting on two qubits.
    This decomposition is not minimal.

    Args:
        circuit (qibo.models.Circuit): circuit to be decomposed.

    Return:
        blocks (list): list of blocks that act on two qubits.
    """
    blocks = []
    all_gates = copy(circuit.queue)
    two_qubit_gates = count_2q_gates(all_gates)
    while two_qubit_gates > 0:
        for idx, gate in enumerate(all_gates):
            if len(gate.qubits) == 2:
                qubits = gate.qubits
                block_gates = find_previous_gates(all_gates[0:idx], qubits)
                block_gates.append(gate)
                block_gates += find_successive_gates(all_gates[idx + 1 :], qubits)
                block = Block(qubits=qubits, gates=block_gates)
                remove_gates(all_gates, block_gates)
                two_qubit_gates -= 1
                blocks.append(block)
                break
            if len(gate.qubits) >= 3:
                raise_error(BlockingError, "Gates targeting more than 2 qubits are not supported.")
    # Now we need to deal with the remaining spare single qubit gates
    while len(all_gates) > 0:
        first_qubit = all_gates[0].qubits[0]
        block_gates = gates_on_qubit(gatelist=all_gates, qubit=first_qubit)
        remove_gates(all_gates, block_gates)
        # Add other single qubits if there are still single qubit gates
        if len(all_gates) > 0:
            second_qubit = all_gates[0].qubits[0]
            second_qubit_block_gates = gates_on_qubit(gatelist=all_gates, qubit=second_qubit)
            block_gates += second_qubit_block_gates
            remove_gates(all_gates, second_qubit_block_gates)
            block = Block(qubits=(first_qubit, second_qubit), gates=block_gates, entangled=False)
        # In case there are no other spare single qubit gates create a block using a following qubit as placeholder
        else:
            block = Block(qubits=(first_qubit, (first_qubit + 1) % circuit.nqubits), gates=block_gates, entangled=False)
        blocks.append(block)
    return blocks


def gates_on_qubit(gatelist, qubit):
    """Return a list of all single qubit gates in gatelist acting on a specific qubit."""
    selected_gates = []
    for gate in gatelist:
        if gate.qubits[0] == qubit:
            selected_gates.append(gate)
    return selected_gates


def remove_gates(gatelist, remove_list):
    """Remove all gates present in remove_list from gatelist."""
    for gate in remove_list:
        gatelist.remove(gate)


def count_2q_gates(gatelist: list):
    """Return the number of two qubit gates in a list of gates."""
    return len([gate for gate in gatelist if len(gate.qubits) == 2])


def find_successive_gates(gates: list, qubits: tuple):
    """Return a list containing all gates acting on qubits until a new two qubit gate acting on qubits is found."""
    successive_gates = []
    for qubit in qubits:
        for gate in gates:
            if (len(gate.qubits) == 1) and (gate.qubits[0] == qubit):
                successive_gates.append(gate)
            if (len(gate.qubits) == 2) and (qubit in gate.qubits):
                break
    return successive_gates


def find_previous_gates(gates: list, qubits: tuple):
    """Return a list containing all gates acting on qubits."""
    previous_gates = []
    for gate in gates:
        if gate.qubits[0] in qubits:
            previous_gates.append(gate)
    return previous_gates
