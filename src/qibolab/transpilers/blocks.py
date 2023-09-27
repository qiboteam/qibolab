from copy import copy

import networkx as nx
from qibo import Circuit, gates
from qibo.config import raise_error

from .abstract import find_gates_qubits_pairs

DEBUG = True


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

    def count_2q_gates(self):
        return count_2q_gates(gatelist=self.get_gates())

    def get_qubits(self):
        return tuple(sorted(self.qubits))

    def get_gates(self):
        return self.gates

    def get_name(self):
        return self.name

    def info(self):
        print("Block Name: ", self.get_name())
        print("Qubits: ", self.get_qubits())
        print("Gates: ", self.get_gates())
        print("Number of two qubits gates: ", self.count_2q_gates())
        print("Entangled: ", self.entangled)

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
    if circuit.nqubits < 2:
        raise_error(BlockingError, "Only circuits with at least two qubits can be decomposed with this function.")
    dag = create_dag(circuit)
    initial_blocks = initial_block_deomposition(circuit)
    return initial_blocks


def initial_block_deomposition(circuit: Circuit):
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
    show_dag(dag)
    cleaned_dag = remove_redundant_connections(dag)
    for layer, nodes in enumerate(nx.topological_generations(dag)):
        for node in nodes:
            dag.nodes[node]["layer"] = layer
    show_dag(cleaned_dag)
    return cleaned_dag


def remove_redundant_connections(G):
    """Remove redundant connection from a DAG"""
    # Create a copy of the input DAG
    new_G = G.copy()
    # Iterate through the nodes in topological order
    for node in nx.topological_sort(G):
        # Compute the set of nodes reachable from the current node
        reachable_nodes = set(nx.descendants(new_G, node)) | {node}
        # Remove edges that are redundant
        for neighbor in list(new_G.neighbors(node)):
            if neighbor in reachable_nodes:
                new_G.remove_edge(node, neighbor)
    return new_G


def show_dag(dag):
    """Plot DAG"""
    import matplotlib.pyplot as plt

    pos = nx.multipartite_layout(dag, subset_key="layer")
    fig, ax = plt.subplots()
    nx.draw_networkx(dag, pos=pos, ax=ax)
    ax.set_title("DAG layout in topological order")
    fig.tight_layout()
    plt.show()
