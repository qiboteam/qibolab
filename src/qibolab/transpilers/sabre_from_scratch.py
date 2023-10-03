from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
from qibo import gates
from qibo.models import Circuit

MAX_ITER = 2000
DEBUG = False


def create_circuit_repr(circuit: Circuit):
    """Translate qibo circuit into a list of two qubit gates to be used by the transpiler.

    Args:
        circuit (qibo.models.Circuit): circuit to be transpiled.

    Returns:
        translated_circuit (dict): dict containing qubits targeted by two qubit gates.
    """
    translated_circuit = {}
    for idx, gate in enumerate(circuit.queue):
        if len(gate.qubits) == 2:
            translated_circuit["g" + str(idx)] = tuple(sorted(gate.qubits))
        if len(gate.qubits) >= 3:
            raise ValueError("Gates targeting more than 2 qubits are not supported")
    return translated_circuit


def create_dag(circuit):
    """Create direct acyclic graph (dag) of the circuit based on two qubit gates commutativity relations.

    Args:
        circuit (qibo.models.Circuit): circuit to be transformed into dag.

    Returns:
        cleaned_dag (nx.DiGraph): dag of the circuit.
    """
    circuit = create_circuit_repr(circuit)
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
    return remove_redundant_connections(dag)


def remove_redundant_connections(dag):
    """Remove redundant connection from a DAG unsing transitive reduction."""
    new_dag = nx.DiGraph()
    transitive_reduction = nx.transitive_reduction(dag)
    new_dag.add_edges_from(transitive_reduction.edges)
    return new_dag


class CircuitMap:
    """Class to keep track of the circuit and physical-logical mapping during routing.

    Attributes:
        circuit_repr (dict): two qubit gates (gi) of the circuit represented as {gi: (q0,q1)}.
        _physical_logical (dict): current logical to physical qubit mapping.
        _circuit_logical (list): initial circuit to current logical circuit mapping.
        routed_circuit (dict): current routed circuit.
        _swaps (int): number of added swaps.
    """

    def __init__(self, initial_layout, circuit):
        self._circuit_logical = list(range(len(initial_layout)))
        self._physical_logical = initial_layout
        self.circuit_repr = create_circuit_repr(circuit)
        self.routed_circuit = {}
        self._swaps = 0

    def info(self):
        """Return circuit information"""
        info = {}
        info["circuit_logical"] = self._circuit_logical
        info["physical_logical"] = self._physical_logical
        info["added_swaps"] = self._swaps
        info["routed_circuit"] = self.routed_circuit
        return info

    def execute_gate(self, gate):
        """Execute a gate by removing it from the circuit representation
        and adding it to the routed circuit.
        """
        self.routed_circuit[gate] = self.circuit_to_logical(self.circuit_repr[gate])
        del self.circuit_repr[gate]

    def qibo_circuit(self):
        """Return qibo circuit of the routed circuit (using CNOT gates)."""
        qibo_circuit = Circuit(len(self._circuit_logical))
        for gate, qubits in self.routed_circuit.items():
            gate_type = "".join(i for i in gate if not i.isdigit())
            if gate_type == "swap":
                qibo_circuit.add(gates.SWAP(*qubits))
            else:
                qibo_circuit.add(gates.CNOT(*qubits))
        return qibo_circuit

    def update(self, swap):
        """Update the logical-physical qubit mapping after applying a SWAP
        and add the gate to the routed circuit.
        """
        self.routed_circuit["swap" + str(self._swaps)] = swap
        self._swaps += 1
        idx_0, idx_1 = self._circuit_logical.index(swap[0]), self._circuit_logical.index(swap[1])
        self._circuit_logical[idx_0], self._circuit_logical[idx_1] = swap[1], swap[0]

    def get_logical_qubits(self, gate):
        """Return the current logical qubits where a gate is acting"""
        return self.circuit_to_logical(self.circuit_repr[gate])

    def get_circuit_qubits(self, gate):
        """Return the initial circuit qubits where a gate is acting"""
        return self.circuit_repr[gate]

    def get_physical_qubits(self, gate, string=True):
        """Return the physical qubits where a gate is acting.
        If string is True return in the form ("qi", "qj").
        If string is False return them in the form (i, j).
        """
        physical_qubits = self.logical_to_physical(self.get_logical_qubits(gate))
        if string is True:
            return physical_qubits
        return self.string_to_int(physical_qubits)

    def logical_to_physical(self, logical_qubits):
        """Return the physical qubits associated to the logical qubits."""
        physical = []
        for i in range(2):
            physical += [k for k, v in self._physical_logical.items() if v is logical_qubits[i]]
        return tuple(physical)

    def circuit_to_logical(self, circuit_qubits):
        """Return the physical qubits associated to the logical qubits."""
        return tuple(self._circuit_logical[circuit_qubits[i]] for i in range(2))

    @staticmethod
    def string_to_int(qubits):
        """Convert qubits written in the form ("qi", "qj") to the form (i,j)"""
        return tuple(int(qubits[i].replace("q", "")) for i in range(2))


class Sabre:
    def __init__(self, connectivity: nx.Graph, lookahead: int = 2, decay: float = 0.6):
        """SabreSwap initializer.

        Args:
            connectivity (dict): hardware chip connectivity.
            lookahead (int): lookahead factor, how many dag layers will be considered in computing the cost.
            decay (float): value in interval [0,1].
                How the weight of the distance in the dag layers decays in computing the cost.
        """
        self.connectivity = connectivity
        self.lookahead = lookahead
        self.decay = decay
        self._dist_matrix = None
        self._dag = None
        self._front_layer = None
        self.circuit = None
        self._memory_map = None

    def update_front_layer(self):
        """Update the front layer of the dag."""
        self._front_layer = self.get_dag_layer(0)

    def get_dag_layer(self, n_layer):
        """Return the n topological layer of the dag."""
        layer_nodes = []
        for layer, nodes in enumerate(nx.topological_generations(self._dag)):
            for node in nodes:
                self._dag.nodes[node]["layer"] = layer
                if layer == n_layer:
                    layer_nodes.append(node)
        return layer_nodes

    def added_swaps(self):
        """Return the number of SWAP gates added to the circuit during routing"""
        return self.circuit._swaps

    def __call__(self, circuit, initial_layout):
        """Route the circuit.

        Args:
            circuit (qibo.models.Circuit): circuit to be routed.
            initial_layout (dict): initial physical to logical qubit mapping.

        Returns:
            routed circuit representation.
        """
        self.preprocessing(circuit=circuit, initial_layout=initial_layout)
        self.update_front_layer()
        i = 0
        while self._dag.number_of_nodes() != 0:
            i += 1
            if i == MAX_ITER:
                print("Transpiling exit because reched max iter")
                break
            execute_gate_list = self.check_execution()
            if execute_gate_list is not None:
                self.execute_gates(execute_gate_list)
            else:
                self.find_new_mapping()
        return self.circuit.routed_circuit, self.circuit.qibo_circuit()

    def find_new_mapping(self):
        """Find the new best mapping by adding one swap."""
        candidates_evaluation = {}
        self._memory_map.append(deepcopy(self.circuit._circuit_logical))
        for candidate in self.swap_candidates():
            candidates_evaluation[candidate] = self.compute_cost(candidate)
        best_cost = min(candidates_evaluation.values())
        best_candidate = list(candidates_evaluation.keys())[list(candidates_evaluation.values()).index(best_cost)]
        self.circuit.update(best_candidate)

    def compute_cost(self, candidate):
        """Compute the cost associated to a possible SWAP candidate."""
        temporary_circuit = deepcopy(self.circuit)
        temporary_circuit.update(candidate)
        if not self.check_new_mapping(temporary_circuit._circuit_logical):
            return float("inf")
        tot_distance = 0.0
        weight = 1.0
        for layer in range(self.lookahead + 1):
            layer_gates = self.get_dag_layer(layer)
            avg_layer_distance = 0.0
            for gate in layer_gates:
                qubits = temporary_circuit.get_physical_qubits(gate, string=False)
                avg_layer_distance += (self._dist_matrix[qubits[0], qubits[1]] - 1.0) / len(layer_gates)
            # tot_distance += (decay^n_layer)*average_layer_distance
            tot_distance += weight * avg_layer_distance
            weight *= self.decay
        return tot_distance

    def check_new_mapping(self, map):
        """Check that the candidate will generate a new qubit mapping in order to avoid ending up in infinite cycles.
        If the mapping is not new the cost associated to that candidate will be infinite."""
        if map in self._memory_map:
            return False
        return True

    def swap_candidates(self):
        """Return a list of possible candidate SWAPs (to be applied on logical qubits directly).
        The possible candidates are the ones sharing at least one qubit with a gate in the front layer.
        """
        candidates = []
        for gate in self._front_layer:
            qubits = self.circuit.get_physical_qubits(gate)
            for qubit in qubits:
                connected_qubits = self.connectivity.neighbors(qubit)
                for connected in connected_qubits:
                    candidate = tuple(
                        sorted((self.circuit._physical_logical[qubit], self.circuit._physical_logical[connected]))
                    )
                    if candidate not in candidates:
                        candidates.append(candidate)
        return candidates

    def preprocessing(self, circuit: Circuit, initial_layout):
        """The following objects will be initialised:
        - circuit: class to represent circuit and to perform logical-physical qubit mapping.
        - _dist_matrix: matrix reporting the shortest path lengh between all node pairs.
        - _dag: direct acyclic graph of the circuit based on commutativity.
        """
        self.circuit = CircuitMap(initial_layout, circuit)
        self._dist_matrix = nx.floyd_warshall_numpy(self.connectivity)
        self._dag = create_dag(self.circuit.circuit_repr)
        self._memory_map = []

    def check_execution(self):
        """Check if some gates in the front layer can be executed in the current configuration.

        Returns:
            list of executable gates if there are, None otherwise.
        """
        gatelist = []
        for gate in self._front_layer:
            qubits = self.circuit.get_physical_qubits(gate)
            if qubits in self.connectivity.edges:
                gatelist.append(gate)
        if len(gatelist) == 0:
            return None
        return gatelist

    def execute_gates(self, gatelist: list):
        """Execute gates: remove the correspondent nodes from the dag and circuit representation.
        The executed gates will be added to the routed circuit. Then update the front layer of the dag.
        Reset the mapping memory.
        """
        for gate in gatelist:
            self.circuit.execute_gate(gate)
            self._dag.remove_node(gate)
        self.update_front_layer()
        self._memory_map = []

    def dag_depth(self):
        """Return the actual lenght of the dag as the longest path."""
        return nx.dag_longest_path_length(self._dag)

    def draw_circuit_dag(self, filename=None):
        """Draw the direct acyclic graph of circuit in topological order.

        Args:
            filename (str): name of the saved image, if None the image will be showed.
        """
        pos = nx.multipartite_layout(self._dag, subset_key="layer")
        fig, ax = plt.subplots()
        nx.draw_networkx(self._dag, pos=pos, ax=ax)
        ax.set_title("DAG layout in topological order")
        fig.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
