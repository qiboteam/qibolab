import random
from enum import Enum, auto

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from more_itertools import pairwise
from qibo import gates
from qibo.config import log, raise_error
from qibo.models import Circuit

DEFAULT_INIT_SAMPLES = 100


class QubitInitMethod(Enum):
    """A class to define the initial qubit mapping methods."""

    greedy = auto()
    subgraph = auto()
    custom = auto()


class Transpiler:
    """A class to perform initial qubit mapping and connectivity matching.

    Properties:
        connectivity (networkx.graph): chip connectivity.
        init_method (string or QubitInitMethod): initial qubit mapping method.
        init_samples (int): number of random qubit initializations for greedy initial qubit mapping.

    Attributes:
        _circuit_repr (list): quantum circuit represented as a list (only 2 qubit gates).
        _mapping (dict): circuit to physical qubit mapping during transpiling.
        _graph (networkx.graph): qubit mapped as nodes of the connectivity graph.
        _qubit_map (np.array): circuit to physical qubit mapping during transpiling as vector.
        _circuit_position (int): position in the circuit.
        _added_swaps (int): number of swaps added to the circuit to match connectivity.
    """

    def __init__(self, connectivity, init_method="greedy", init_samples=None):
        self.connectivity = connectivity
        self.init_method = init_method
        if self.init_method is QubitInitMethod.greedy and init_samples is None:
            init_samples = DEFAULT_INIT_SAMPLES
        self.init_samples = init_samples

        self._circuit_repr = None
        self._mapping = None
        self._graph = None
        self._qubit_map = None
        self._circuit_position = 0
        self._added_swaps = 0

    def transpile(self, qibo_circuit):
        """Qubit mapping initialization and circuit transpiling.

        Args:
            qibo_circuit (qibo.Circuit): circuit to be transpiled.

        Returns:
            hardware_mapped_circuit (qibo.Circuit): circut mapped to hardware topology.
            final_mapping (dict): final qubit mapping.
            init_mapping (dict): initial qubit mapping.
            added_swaps (int): number of swap gates added.
        """
        self._circuit_position = 0
        self._added_swaps = 0
        self.translate_circuit(qibo_circuit)
        keys = list(self._connectivity.nodes())
        if self._init_method is QubitInitMethod.greedy:
            self.greedy_init()
        elif self._init_method is QubitInitMethod.subgraph:
            self.subgraph_init()
        elif self._init_method is QubitInitMethod.custom:
            self._mapping = dict(zip(keys, self._mapping.values()))
            self._graph = nx.relabel_nodes(self._connectivity, self._mapping)
        # Inverse permutation
        init_qubit_map = np.argsort(list(self._mapping.values()))
        init_mapping = dict(zip(keys, init_qubit_map))
        self._qubit_map = np.sort(init_qubit_map)
        self.init_circuit(qibo_circuit)
        self.first_transpiler_step(qibo_circuit)
        while len(self._circuit_repr) != 0:
            self.transpiler_step(qibo_circuit)
        final_mapping = {key: init_qubit_map[self._qubit_map[i]] for i, key in enumerate(keys)}
        return (
            self.init_mapping_circuit(self.transpiled_circuit, init_qubit_map),
            final_mapping,
            init_mapping,
            self._added_swaps,
        )

    def transpiler_step(self, qibo_circuit):
        """Transpilation step. Find new mapping, add swap gates and apply gates that can be run with this configuration.

        Args:
            qibo_circuit (qibo.Circuit): circuit to be transpiled.
        """
        len_2q_circuit = len(self._circuit_repr)
        path, meeting_point = self.relocate()
        self.add_swaps(path, meeting_point)
        self.update_qubit_map()
        self.add_gates(qibo_circuit, len_2q_circuit - len(self._circuit_repr))

    def first_transpiler_step(self, qibo_circuit):
        """First transpilation step. Apply gates that can be run with the initial qubit mapping.

        Args:
            qibo_circuit (qibo.Circuit): circuit to be transpiled.
        """
        len_2q_circuit = len(self._circuit_repr)
        self._circuit_repr = self.reduce(self._graph)
        self.add_gates(qibo_circuit, len_2q_circuit - len(self._circuit_repr))

    def custom_qubit_mapping(self, map):
        """Define a custom initial qubit mapping.

        Args:
            map (list): List reporting the circuit to chip qubit mapping,
            example [1,2,0] to assign the logical to physical qubit mapping.
        """
        self.init_method = QubitInitMethod.custom
        self._mapping = dict(zip(range(len(map)), map))

    @property
    def connectivity(self):
        return self._connectivity

    @connectivity.setter
    def connectivity(self, connectivity):
        """Set the hardware chip connectivity.

        Args:
            connectivity (networkx graph): define connectivity.
        """

        if isinstance(connectivity, nx.Graph):
            self._connectivity = connectivity
        else:
            raise_error(TypeError, "Use networkx graph for custom connectivity")

    def draw_connectivity(self):  # pragma: no cover
        """Show connectivity graph."""
        pos = nx.spectral_layout(self._connectivity)
        nx.draw(self._connectivity, pos=pos, with_labels=True)
        plt.show()

    @property
    def init_method(self):
        return self._init_method

    @init_method.setter
    def init_method(self, init_method, init_samples=100):
        """Set the initial mapping method for the transpiler.

        Args:
            init_method (string): Initial mapping method ("greedy" or "subgraph").
        """
        if isinstance(init_method, str):
            init_method = QubitInitMethod[init_method]
        self._init_method = init_method

    @property
    def init_samples(self):
        return self._init_samples

    @init_samples.setter
    def init_samples(self, init_samples):
        """Set the initial mapping method for the transpiler.

        Args:
            init_samples (int): Number of random qubit mapping samples to be tested
            (required only for "greedy" initialization).
        """
        if init_samples is not None:
            self.init_method = QubitInitMethod.greedy
        self._init_samples = init_samples

    def translate_circuit(self, qibo_circuit):
        """Translate qibo circuit into a list of two qubit gates to be used by the transpiler.

        Args:
            qibo_circuit (qibo.Circuit): circuit to be transpiled.
        """
        translated_circuit = []
        for index, gate in enumerate(qibo_circuit.queue):
            if len(gate.qubits) == 2:
                gate_qubits = list(gate.qubits)
                gate_qubits.sort()
                gate_qubits.append(index)
                translated_circuit.append(gate_qubits)
            if len(gate.qubits) >= 3:
                raise_error(ValueError, "ERROR do not use gates acting on more than 2 qubits")
        self._circuit_repr = translated_circuit

    def reduce(self, graph):
        """Reduce the circuit, delete a 2-qubit gate if it can be applied on the current configuration.

        Args:
            graph (networkx.Graph): current hardware qubit mapping.

        Returns:
            new_circuit (list): reduced circuit.
        """
        new_circuit = self._circuit_repr.copy()
        while new_circuit != [] and (new_circuit[0][0], new_circuit[0][1]) in graph.edges():
            del new_circuit[0]
        return new_circuit

    def subgraph_init(self):
        """Subgraph isomorphism initialization, NP-complete it can take a long time for large circuits."""
        H = nx.Graph()
        H.add_nodes_from([i for i in range(self._connectivity.number_of_nodes())])
        GM = nx.algorithms.isomorphism.GraphMatcher(self._connectivity, H)
        i = 0
        H.add_edge(self._circuit_repr[i][0], self._circuit_repr[i][1])
        while GM.subgraph_is_monomorphic() == True:
            result = GM
            i = i + 1
            H.add_edge(self._circuit_repr[i][0], self._circuit_repr[i][1])
            GM = nx.algorithms.isomorphism.GraphMatcher(self._connectivity, H)
            if self._connectivity.number_of_edges() == H.number_of_edges() or i == len(self._circuit_repr) - 1:
                G = nx.relabel_nodes(self._connectivity, result.mapping)
                self._graph = G
                self._mapping = result.mapping
                return
        G = nx.relabel_nodes(self._connectivity, result.mapping)
        self._graph = G
        self._mapping = result.mapping

    def greedy_init(self):
        """Initialize the circuit with greedy algorithm let a maximum number of 2-qubit
        gates can be applied without introducing any SWAP gate"""
        nodes = self._connectivity.number_of_nodes()
        keys = list(self._connectivity.nodes())
        final_mapping = dict(zip(keys, list(range(nodes))))
        final_graph = nx.relabel_nodes(self._connectivity, final_mapping)
        final_cost = len(self.reduce(final_graph))
        for _ in range(self._init_samples):
            mapping = dict(zip(keys, random.sample(range(nodes), nodes)))
            graph = nx.relabel_nodes(self._connectivity, mapping)
            cost = len(self.reduce(graph))
            if cost == 0:
                self._graph = graph
                self._mapping = mapping
                return
            if cost < final_cost:
                final_graph = graph
                final_mapping = mapping
                final_cost = cost
        self._graph = final_graph
        self._mapping = final_mapping

    def map_list(self, path):
        """Return all possible walks of qubits for a given path.

        Args:
            path (list): path to move qubits.

        Returns:
            mapping_list (list): all possible walks of qubits for a given path.
            meeting_point_list (list): all possible qubit meeting point in the path.
        """
        path_ends = [path[0], path[-1]]
        path_middle = path[1:-1]
        mapping_list = []
        meeting_point_list = []
        for i in range(len(path) - 1):
            values = path_middle[:i] + path_ends + path_middle[i:]
            mapping = dict(zip(path, values))
            mapping_list.append(mapping)
            meeting_point_list.append(i)
        return mapping_list, meeting_point_list

    def relocate(self):
        """A small greedy algorithm to decide which path to take, and how qubits should walk.

        Returns:
            final_path (list): best path to move qubits.
            meeting_point (int): qubit meeting point in the path.
        """
        nodes = self._graph.number_of_nodes()
        circuit = self.reduce(self._graph)
        final_circuit = circuit
        keys = list(range(nodes))
        final_graph = self._graph
        final_mapping = dict(zip(keys, keys))
        # Consider all shortest paths
        path_list = [p for p in nx.all_shortest_paths(self._graph, source=circuit[0][0], target=circuit[0][1])]
        self._added_swaps += len(path_list[0]) - 2
        final_path = path_list[0]
        # Reduce the number of paths to be faster
        for path in path_list:
            list_, meeting_point_list = self.map_list(path)
            for j, mapping in enumerate(list_):
                new_graph = nx.relabel_nodes(self._graph, mapping)
                new_circuit = self.reduce(new_graph)
                # Greedy looking for the optimal path and the optimal walk on this path
                if len(new_circuit) < len(final_circuit):
                    final_graph = new_graph
                    final_circuit = new_circuit
                    final_mapping = mapping
                    final_path = path
                    meeting_point = meeting_point_list[j]
        self._graph = final_graph
        self._mapping = final_mapping
        self._circuit_repr = final_circuit
        return final_path, meeting_point

    def init_circuit(self, qibo_circuit):
        """Initialize the transpiled circuit

        Args:
            Args: qibo_circuit (qibo.Circuit): circuit to be transpiled.
        """
        nodes = self._connectivity.number_of_nodes()
        qubits = qibo_circuit.nqubits
        if qubits > nodes:
            raise_error(ValueError, "There are not enough physical qubits in the hardware to map the circuit")
        elif qubits == nodes:
            new_circuit = Circuit(nodes)
        else:
            log.warning(
                "You are using more physical qubits than required by the circuit, some qubits will be added to the circuit"
            )
            new_circuit = Circuit(nodes)
        self.transpiled_circuit = new_circuit

    def init_mapping_circuit(self, circuit, qubit_map):
        """Initial qubit mapping of the transpiled qibo circuit

        Args:
            circuit (qibo.Circuit): transpiled qibo circuit.
            qubit_map (np.array): initial qubit mapping.

        Returns:
            new_circuit (qibo.Circuit): transpiled circuit mapped with initial qubit mapping.
        """
        new_circuit = Circuit(self._connectivity.number_of_nodes())
        for gate in circuit.queue:
            new_circuit.add(gate.on_qubits({q: qubit_map[q] for q in gate.qubits}))
        return new_circuit

    def add_gates(self, qibo_circuit, matched_gates):
        """Add one and two qubit gates to transpiled circuit until connectivity is matched

        Args:
            qibo_circuit (qibo.Circuit): circuit to be transpiled.
            matched_gates (int): number of two qubit gates that can be applied with the current qubit mapping
        """
        index = 0
        while self._circuit_position < len(qibo_circuit.queue):
            gate = qibo_circuit.queue[self._circuit_position]
            if len(gate.qubits) == 1:
                self.transpiled_circuit.add(gate.on_qubits({gate.qubits[0]: self._qubit_map[gate.qubits[0]]}))
                self._circuit_position += 1
            else:
                index += 1
                if index == matched_gates + 1:
                    break
                else:
                    self.transpiled_circuit.add(
                        gate.on_qubits(
                            {
                                gate.qubits[0]: self._qubit_map[gate.qubits[0]],
                                gate.qubits[1]: self._qubit_map[gate.qubits[1]],
                            }
                        )
                    )
                    self._circuit_position += 1

    def add_swaps(self, path, meeting_point):
        """Add swaps to the transpiled circuit to move qubits

        Args:
            path (list): path to move qubits.
            meeting_point (int): qubit meeting point in the path.
        """
        forward = path[0 : meeting_point + 1]
        backward = path[meeting_point + 1 :: -1]
        if len(forward) > 1:
            for f1, f2 in pairwise(forward):
                self.transpiled_circuit.add(gates.SWAP(self._qubit_map[f1], self._qubit_map[f2]))
        if len(backward) > 1:
            for b1, b2 in pairwise(backward):
                self.transpiled_circuit.add(gates.SWAP(self._qubit_map[b1], self._qubit_map[b2]))

    def update_qubit_map(self):
        """Update the qubit mapping after adding swaps"""
        old_mapping = self._qubit_map.copy()
        for key, value in self._mapping.items():
            self._qubit_map[value] = old_mapping[key]
