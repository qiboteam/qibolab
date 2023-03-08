import random
from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sympy
from qibo import gates
from qibo.config import log, raise_error
from qibo.models import Circuit


class Transpiler:
    def __init__(self, connectivity, init_method="greedy", init_samples=100):
        self.connectivity = connectivity
        self._init_method = init_method
        self._init_samples = init_samples

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
        if self._init_method == "greedy":
            self.greedy_init()
        elif self._init_method == "subgraph":
            self.subgraph_init()
        elif self._init_method == "custom":
            self._mapping = {keys[i]: self._mapping["q" + str(i)] for i in range(len(keys))}
            self._graph = nx.relabel_nodes(self._connectivity, self._mapping)
        else:
            raise_error(ValueError, "Wrong qubit mapping initialization method (use 'greedy', 'subgraph' or 'custom')")
        init_qubit_map = self.init_qubit_map()
        init_mapping = {keys[i]: init_qubit_map[i] for i in range(len(keys))}
        self._qubit_map = np.sort(init_qubit_map)
        self.init_circuit(qibo_circuit)
        len_2q_circuit = len(self._circuit_repr)
        self._circuit_repr = self.reduce(self._graph)
        matched_2q_gates = len_2q_circuit - len(self._circuit_repr)
        self.add_gates(qibo_circuit, matched_2q_gates)
        while len(self._circuit_repr) != 0:
            len_2q_circuit = len(self._circuit_repr)
            path, meeting_point = self.relocate()
            matched_2q_gates = len_2q_circuit - len(self._circuit_repr)
            self.add_swaps(path, meeting_point)
            self.update_qubit_map()
            self.add_gates(qibo_circuit, matched_2q_gates)
        final_mapping = {keys[i]: init_qubit_map[self._qubit_map[i]] for i in range(len(keys))}
        return (
            self.init_mapping_circuit(self.transpiled_circuit, init_qubit_map),
            final_mapping,
            init_mapping,
            self._added_swaps,
        )

    def custom_qubit_mapping(self, map):
        """Define a custom initial qubit mapping.

        Args:
            map (dict): Dictionary of the type {q0: 1, q1: 2, q2: 0} reporting the circuit to chip qubit mapping.
        """
        self._init_method = "custom"
        self._mapping = map

    @property
    def connectivity(self):
        return self._connectivity

    @connectivity.setter
    def connectivity(self, connectivity):
        """Set the hardware chip connectivity.

        Args:
            connectivity (string or networkx graph): set a special TII chip connectivity (using a string) or define a
            custom connectivity (using a networkx graph).
        """

        if isinstance(connectivity, type(nx.Graph())):
            self._connectivity = connectivity
        else:
            raise_error(TypeError, "Use networkx graph for custom connectivity")

    def draw_connectivity(self):
        """Show connectivity graph."""
        pos = nx.spectral_layout(self._connectivity)
        nx.draw(self._connectivity, pos=pos, with_labels=True)
        plt.show()

    def init_method(self, init_method, init_samples=100):
        """Set the initial mapping method for the transpiler.

        Args:
            init_method (string): Initial mapping method ("greedy" or "subgraph").
            init_samples (int): Number of random qubit mapping samples to be tested
            (required only for "greedy" initialization).
        """
        self._init_method = init_method
        self._init_samples = init_samples

    def translate_circuit(self, qibo_circuit):
        """Translate qibo circuit into a list of two qubit gates to be used by the transpiler.

        Args:
            qibo_circuit (qibo.Circuit): circuit to be transpiled.
        """
        translated_circuit = []
        index = 1
        for gate in qibo_circuit.queue:
            if len(gate.qubits) == 2:
                gate_qubits = list(gate.qubits)
                gate_qubits.sort()
                gate_qubits.append(sympy.symbols(f"g{index}"))
                translated_circuit.append(gate_qubits)
                index += 1
            if len(gate.qubits) >= 3:
                raise_error("ERROR do not use gates acting on more than 2 qubits")
        self._circuit_repr = translated_circuit

    def reduce(self, graph):
        """Reduce the circuit, delete a 2-qubit gate if it can be applied on the current configuration.

        Args:
            graph (networkx.Graph): current hardware qubit mapping.

        Returns:
            new_circuit (list): reduced circuit.
        """
        new_circuit = deepcopy(self._circuit_repr)
        while new_circuit != [] and (new_circuit[0][0], new_circuit[0][1]) in graph.edges():
            del new_circuit[0]
        return new_circuit

    def subgraph_init(self):
        """Subgraph isomorphism initialization, NP-complete it can take a long time for large circuits."""
        H = nx.Graph()
        H.add_nodes_from([i for i in range(0, self._connectivity.number_of_nodes())])
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
        """initialize the circuit with greedy algorithm let a maximum number of 2-qubit
        gates can be applied without introducing any SWAP gate"""
        nodes = self._connectivity.number_of_nodes()
        keys = list(self._connectivity.nodes())
        values = [i for i in range(nodes)]
        final_mapping = {keys[i]: values[i] for i in range(len(keys))}
        final_graph = nx.relabel_nodes(self._connectivity, final_mapping)
        final_cost = len(self.reduce(final_graph))
        for _ in range(self._init_samples):
            random.shuffle(values)
            mapping = {keys[i]: values[i] for i in range(len(keys))}
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

    def init_qubit_map(self):
        """Initial circuit-hardware qubit mapping.

        Returns:
            qubit_map (np.array): array containing the initial qubit mapping of the circuit.
        """
        qubit_map = np.zeros((len(self._mapping.keys()),), dtype=int)
        i = 0
        for key in self._mapping.keys():
            qubit_map[self._mapping[key]] = i
            i += 1
        return qubit_map

    def map_list(self, path):
        """Return all possible walks of qubits for a given path.

        Args:
            path (list): path to move qubits.

        Returns:
            mapping_list (list): all possible walks of qubits for a given path.
            meeting_point_list (list): all possible qubit meeting point in the path.
        """
        keys = path
        path_ends = [path[0]] + [path[-1]]
        path_middle = path[1:-1]
        mapping_list = []
        meeting_point_list = []
        for i in range(len(path) - 1):
            values = path_middle[:i] + path_ends + path_middle[i:]
            mapping = {keys[i]: values[i] for i in range(len(keys))}
            mapping_list.append(mapping)
            meeting_point_list.append(i)
        return mapping_list, meeting_point_list

    def relocate(self):
        """A small greedy algorithm to decide which path to take, and how qubits should walk.

        Returns:
            final_path (list): best path to move qubits.
            meeting_point (int): qubit meeting point in the path.
        """
        if len(self._circuit_repr) == 0:
            return 0
        nodes = self._graph.number_of_nodes()
        circuit = self.reduce(self._graph)
        final_circuit = circuit
        keys = [i for i in range(0, nodes)]
        values = keys
        final_graph = self._graph
        final_mapping = {keys[i]: values[i] for i in range(len(keys))}
        # Consider all shortest paths
        path_list = [p for p in nx.all_shortest_paths(self._graph, source=circuit[0][0], target=circuit[0][1])]
        self._added_swaps += len(path_list[0]) - 2
        final_path = path_list[0]
        # Reduce the number of paths to be faster
        for i in range(len(path_list)):
            List, meeting_point_list = self.map_list(path_list[i])
            for j in range(len(List)):
                mapping = List[j]
                new_graph = nx.relabel_nodes(self._graph, mapping)
                new_circuit = self.reduce(new_graph)
                # Greedy looking for the optimal path and the optimal walk on this path
                if len(new_circuit) < len(final_circuit):
                    final_graph = new_graph
                    final_circuit = new_circuit
                    final_mapping = mapping
                    final_path = path_list[i]
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
            if len(gate.qubits) == 1:
                new_circuit.add(
                    gate.on_qubits(
                        {
                            gate.qubits[0]: qubit_map[gate.qubits[0]],
                        }
                    )
                )
            elif len(gate.qubits) == 2:
                new_circuit.add(
                    gate.on_qubits(
                        {gate.qubits[0]: qubit_map[gate.qubits[0]], gate.qubits[1]: qubit_map[gate.qubits[1]]}
                    )
                )
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
        backward = path[meeting_point + 1 :]
        if len(forward) > 1:
            for i in range(len(forward) - 1):
                self.transpiled_circuit.add(gates.SWAP(self._qubit_map[forward[i]], self._qubit_map[forward[i + 1]]))
        if len(backward) > 1:
            for i in range(len(backward) - 1, 0, -1):
                self.transpiled_circuit.add(gates.SWAP(self._qubit_map[backward[i]], self._qubit_map[backward[i - 1]]))

    def update_qubit_map(self):
        """Update the qubit mapping after adding swaps"""
        old_mapping = deepcopy(self._qubit_map)
        for key in self._mapping.keys():
            self._qubit_map[self._mapping[key]] = old_mapping[key]
