from copy import copy

import networkx as nx
from qibo import Circuit, gates
from qibo.config import raise_error

from .abstract import find_gates_qubits_pairs


class Block:
    """A block contains a subset of gates acting on two qubits.

    Args:
        qubits (tuple): qubits where the block is acting.
        gates (list): list of gates that compose the block.
        name (str): name of the block.
        entangled (bool): true if there is at least a two qubit gate in the block.
    """

    def __init__(self, qubits: tuple, gates: list, name: str = None, entangled: bool = True):
        self.qubits = qubits
        self.gates = gates
        self.name = name
        self.entangled = entangled

    def rename(self, name):
        self.name = name

    def add_gate(self, gate: gates.Gate):
        self.gates.append(gate)
        if len(gate.qubits) == 2:
            self.entangled = True

    def get_qubits(self):
        return self.qubits

    def get_gates(self):
        return self.gates

    def get_name(self):
        return self.name

    # TODO
    def kak_decompose(self):
        """Return KAK decomposition of the block.
        This should be done only if the block is entangled.
        """
        raise_error(NotImplementedError)


def block_decomposition(circuit: Circuit):
    """Decompose a circuit into blocks of gates acting on two qubits.

    Args:
        circuit (qibo.models.Circuit): circuit to be decomposed.

    Return:
        blocks (list): list of blocks that act on two qubits.
    """
    dag = create_dag(circuit)
    pairs = find_gates_qubits_pairs(circuit)


def initial_block_deomposition(circuit):
    """Decompose a circuit into blocks of gates acting on two qubits.
    This decomposition is not minimal.

    Args:
        circuit (qibo.models.Circuit): circuit to be decomposed.

    Return:
        blocks (list): list of blocks that act on two qubits.
    """
    blocks = []
    all_gates = copy(circuit.queue)
    # while len(all_gates)>3:
    for idx, gate in enumerate(all_gates):
        if len(gate.qubits) == 2:
            qubits = gate.qubits
            block_gates = find_previous_gates(all_gates[0:idx], qubits)
            block_gates.append(gate)
            print(block_gates)
            block = Block(qubits=qubits, gates=block_gates)
            for gate in block_gates:
                all_gates.remove(gate)
            blocks.append(block)
            break
        if len(gate.qubits) >= 3:
            raise_error(ValueError, "Gates targeting more than 2 qubits are not supported")
    print(all_gates)
    return blocks


def find_previous_gates(gates: list, qubits: tuple):
    """Return a list containing all gates acting on qubits."""
    previous_gates = []
    for gate in gates:
        if gate.qubits[0] in qubits:
            previous_gates.append(gate)
    return previous_gates


def create_dag(circuit):
    """Create direct acyclic graph (dag) of the circuit based on two qubit gates commutativity relations.

    Args:
        circuit (qibo.models.Circuit): circuit to be transformed into dag.

    Returns:
        cleaned_dag (nx.DiGraph): dag of the circuit.
    """
    circuit = find_gates_qubits_pairs(circuit)
    dag = nx.DiGraph()
    dag.add_nodes_from(list(i for i in range(len(circuit))))
    # Find all successors
    connectivity_list = []
    for idx, gate in enumerate(circuit):
        for next_idx, next_gate in enumerate(circuit[idx + 1 :]):
            for qubit in gate:
                if qubit in next_gate:
                    connectivity_list.append((idx, next_idx + idx + 1))
    dag.add_edges_from(connectivity_list)
    for layer, nodes in enumerate(nx.topological_generations(dag)):
        for node in nodes:
            dag.nodes[node]["layer"] = layer
    # Remove redundant connections
    cleaned_dag = dag.copy()
    for node in dag.nodes:
        for successor in dag.successors(node):
            if not dag.nodes[successor]["layer"] == dag.nodes[node]["layer"] + 1:
                cleaned_dag.remove_edge(node, successor)
    return cleaned_dag
