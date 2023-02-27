import random
from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sympy
from qibo import gates
from qibo.models import Circuit


class Transpiler:
    def __init__(self, connectivity="21_qubits", init_method="greedy", init_samples=100):
        self.connectivity = self.set_connectivity(connectivity)
        self.init_method = init_method
        self.init_samples = init_samples

    def custom_qubit_mapping(self, map):
        """Define initial qubit mapping using a dictionary of the type {q0: 1, q1: 2, q2: 0}"""
        self.init_method = "custom"
        self.mapping = map

    def set_connectivity(self, connectivity):
        if isinstance(connectivity, str):
            return self.set_special_connectivity(connectivity)
        elif isinstance(connectivity, type(nx.Graph())):
            return connectivity
        else:
            print("Error: use networkx graph to define connectivity; 21 qubit chip will be used as connectivity")
            return self.set_special_connectivity("21_qubits")

    def draw_connectivity(self):
        """Show connectivity graph"""
        pos = nx.spectral_layout(self.connectivity)
        nx.draw(self.connectivity, pos=pos, with_labels=True)
        plt.show()

    def set_init_method(self, init_method, init_samples=100):
        """Set the initial mapping for the transpiler."""
        self.init_method = init_method
        self.init_samples = init_samples

    def translate_circuit(self, qibo_circuit):
        """Translate qibo circuit into a list of two qubit gates to be used by the transpiler"""
        translated_circuit = []
        index = 1
        for gate in qibo_circuit.queue:
            if len(gate.qubits) == 2:
                gate_qubits = [gate.qubits[0], gate.qubits[1]]
                gate_qubits.sort()
                gate_qubits.append(sympy.symbols(f"g{index}"))
                translated_circuit.append(gate_qubits)
                index += 1
            if len(gate.qubits) >= 3:
                print("ERROR do not use gates acting on more than 2 qubits")
                return None
        return translated_circuit

    def clean(self):
        """Remove the second of two adjacent 2-qubit gates (acting on the same qubits)"""
        new_circuit = []
        new_circuit.append(self.circuit_repr[0])
        for i in range(1, len(self.circuit_repr)):
            if not (
                self.circuit_repr[i][0] == self.circuit_repr[i - 1][0]
                and self.circuit_repr[i][1] == self.circuit_repr[i - 1][1]
            ):
                new_circuit.append(self.circuit_repr[i])
        return new_circuit

    def reduce(self, graph):
        """Reduce the circuit, delete a 2-qubit gate if it can be applied on the current configuration"""
        new_circuit = deepcopy(self.circuit_repr)
        while new_circuit != [] and (new_circuit[0][0], new_circuit[0][1]) in graph.edges():
            del new_circuit[0]
        return new_circuit

    def subgraph_init(self):
        """Subgraph isomorphism initialization, NP-complete"""
        H = nx.Graph()
        H.add_nodes_from([i for i in range(0, self.connectivity.number_of_nodes())])
        GM = nx.algorithms.isomorphism.GraphMatcher(self.connectivity, H)
        i = 0
        H.add_edge(self.circuit_repr[i][0], self.circuit_repr[i][1])
        while GM.subgraph_is_monomorphic() == True:
            result = GM
            i = i + 1
            H.add_edge(self.circuit_repr[i][0], self.circuit_repr[i][1])
            GM = nx.algorithms.isomorphism.GraphMatcher(self.connectivity, H)
            if self.connectivity.number_of_edges() == H.number_of_edges() or i == len(self.circuit_repr) - 1:
                G = nx.relabel_nodes(self.connectivity, result.mapping)
                return G, result.mapping
        G = nx.relabel_nodes(self.connectivity, result.mapping)
        return G, result.mapping

    def greedy_init(self):
        """initialize the circuit with greedy algorithm let a maximum number of 2-qubit gates can be applied without introducing any SWAP gate"""
        nodes = self.connectivity.number_of_nodes()
        keys = list(self.connectivity.nodes())
        values = [i for i in range(nodes)]
        final_mapping = {keys[i]: values[i] for i in range(len(keys))}
        final_graph = nx.relabel_nodes(self.connectivity, final_mapping)
        final_cost = len(self.reduce(final_graph))
        for _ in range(self.init_samples):
            random.shuffle(values)
            mapping = {keys[i]: values[i] for i in range(len(keys))}
            graph = nx.relabel_nodes(self.connectivity, mapping)
            cost = len(self.reduce(graph))
            if cost == 0:
                return graph, mapping
            if cost < final_cost:
                final_graph = graph
                final_mapping = mapping
                final_cost = cost
        return final_graph, final_mapping

    def init_qubit_map(self, mapping):
        """Initial circuit-hardware qubit mapping"""
        qubit_map = np.zeros((len(mapping.keys()),), dtype=int)
        i = 0
        for key in mapping.keys():
            qubit_map[mapping[key]] = i
            i += 1
        return qubit_map

    def map_list(self, path):
        """Return all possible walks of qubits for a given path"""
        keys = path
        path_ends = [path[0]] + [path[-1]]
        path_middle = path[1:-1]
        List = []
        meeting_point_list = []
        for i in range(len(path) - 1):
            values = path_middle[:i] + path_ends + path_middle[i:]
            mapping = {keys[i]: values[i] for i in range(len(keys))}
            List.append(mapping)
            meeting_point_list.append(i)
        return List, meeting_point_list

    def relocate(self):
        """A small greedy algorithm to decide which path to take, and how qubits should walk"""
        if len(self.circuit_repr) == 0:
            return 0
        nodes = self.graph.number_of_nodes()
        circuit = self.reduce(self.graph)
        final_circuit = circuit
        keys = [i for i in range(0, nodes)]
        values = keys
        final_graph = self.graph
        final_mapping = {keys[i]: values[i] for i in range(len(keys))}
        # Consider all shortest paths
        path_list = [p for p in nx.all_shortest_paths(self.graph, source=circuit[0][0], target=circuit[0][1])]
        self.n_swap += len(path_list[0]) - 2
        final_path = path_list[0]
        # Reduce the number of paths to be faster
        for i in range(len(path_list)):
            List, meeting_point_list = self.map_list(path_list[i])
            for j in range(len(List)):
                mapping = List[j]
                new_graph = nx.relabel_nodes(self.graph, mapping)
                new_circuit = self.reduce(new_graph)
                # Greedy looking for the optimal path and the optimal walk on this path
                if len(new_circuit) < len(final_circuit):
                    final_graph = new_graph
                    final_circuit = new_circuit
                    final_mapping = mapping
                    final_path = path_list[i]
                    meeting_point = meeting_point_list[j]
        self.graph = final_graph
        self.mapping = final_mapping
        self.circuit_repr = final_circuit
        return final_path, meeting_point

    def init_circuit(self, qibo_circuit):
        """Initialize the transpiled circuit"""
        nodes = self.connectivity.number_of_nodes()
        qubits = qibo_circuit.nqubits
        if qubits > nodes:
            print("ERROR, there are not enough physical qubits to map the circuit")
            return None
        elif qubits == nodes:
            new_circuit = Circuit(nodes)
        else:
            print(
                "WARNING, you are using more physical qubits than required by the circuit, some qubits will be added to the circuit"
            )
            new_circuit = Circuit(nodes)
        return new_circuit

    def init_mapping_circuit(self, circuit, qubit_map):
        """Initial qubit mapping of the transpiled qibo circuit"""
        new_circuit = Circuit(self.connectivity.number_of_nodes())
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
        """Add one and two qubit gates to transpiled circuit until connectivity is matched"""
        index = 0
        while self.pos < len(qibo_circuit.queue):
            gate = qibo_circuit.queue[self.pos]
            if len(gate.qubits) == 1:
                self.transpiled_circuit.add(gate.on_qubits({gate.qubits[0]: self.qubit_map[gate.qubits[0]]}))
                self.pos += 1
            else:
                index += 1
                if index == matched_gates + 1:
                    break
                else:
                    self.transpiled_circuit.add(
                        gate.on_qubits(
                            {
                                gate.qubits[0]: self.qubit_map[gate.qubits[0]],
                                gate.qubits[1]: self.qubit_map[gate.qubits[1]],
                            }
                        )
                    )
                    self.pos += 1

    def add_swaps(self, path, meeting_point):
        """Add swaps to the transpiled circuit to move qubits"""
        forward = path[0 : meeting_point + 1]
        backward = path[meeting_point + 1 :]
        if len(forward) > 1:
            for i in range(len(forward) - 1):
                self.transpiled_circuit.add(gates.SWAP(self.qubit_map[forward[i]], self.qubit_map[forward[i + 1]]))
        if len(backward) > 1:
            for i in range(len(backward) - 1, 0, -1):
                self.transpiled_circuit.add(gates.SWAP(self.qubit_map[backward[i]], self.qubit_map[backward[i - 1]]))
        return

    def update_qubit_map(self):
        """Update the qubit mapping after adding swaps"""
        old_mapping = deepcopy(self.qubit_map)
        for key in self.mapping.keys():
            self.qubit_map[self.mapping[key]] = old_mapping[key]

    def transpile(self, qibo_circuit):
        """Qubit initialization and circuit transpiling.

        Args:
            qibo_circuit: circuit to be transpiled.

        Returns:
            hardware_mapped_circuit: circut mapped to hardware topology.
            final_mapping (dict): final qubit mapping.
            init_mapping (dict): initial qubit mapping.
            added_swaps (int): number of swap gates added.
        """
        self.circuit_repr = self.translate_circuit(qibo_circuit)
        keys = list(self.connectivity.nodes())
        if self.init_method == "greedy":
            self.graph, self.mapping = self.greedy_init()
        elif self.init_method == "subgraph":
            self.graph, self.mapping = self.subgraph_init()
        elif self.init_method == "custom":
            self.mapping = {keys[i]: self.mapping["q" + str(i)] for i in range(len(keys))}
            self.graph = nx.relabel_nodes(self.connectivity, self.mapping)
        else:
            print("ERROR")
        init_qubit_map = self.init_qubit_map(self.mapping)
        init_mapping = {keys[i]: init_qubit_map[i] for i in range(len(keys))}
        self.qubit_map = np.sort(init_qubit_map)
        self.transpiled_circuit = self.init_circuit(qibo_circuit)
        len_2q_circuit = len(self.circuit_repr)
        self.circuit_repr = self.reduce(self.graph)
        self.pos = 0
        matched_2q_gates = len_2q_circuit - len(self.circuit_repr)
        self.add_gates(qibo_circuit, matched_2q_gates)
        self.n_swap = 0
        while len(self.circuit_repr) != 0:
            len_2q_circuit = len(self.circuit_repr)
            path, meeting_point = self.relocate()
            matched_2q_gates = len_2q_circuit - len(self.circuit_repr)
            self.add_swaps(path, meeting_point)
            self.update_qubit_map()
            self.add_gates(qibo_circuit, matched_2q_gates)
        hardware_mapped_circuit = self.init_mapping_circuit(self.transpiled_circuit, init_qubit_map)
        final_mapping = {keys[i]: init_qubit_map[self.qubit_map[i]] for i in range(len(keys))}
        return hardware_mapped_circuit, final_mapping, init_mapping, self.n_swap

    def set_special_connectivity(self, connectivity):
        """Set a TII harware connectivity"""
        if connectivity == "21_qubits":
            Q = sympy.symbols([f"q{i}" for i in range(21)])
            chip = nx.Graph()
            chip.add_nodes_from(Q)
            graph_list_h = [
                (Q[0], Q[1]),
                (Q[1], Q[2]),
                (Q[3], Q[4]),
                (Q[4], Q[5]),
                (Q[5], Q[6]),
                (Q[6], Q[7]),
                (Q[8], Q[9]),
                (Q[9], Q[10]),
                (Q[10], Q[11]),
                (Q[11], Q[12]),
                (Q[13], Q[14]),
                (Q[14], Q[15]),
                (Q[15], Q[16]),
                (Q[16], Q[17]),
                (Q[18], Q[19]),
                (Q[19], Q[20]),
            ]
            graph_list_v = [
                (Q[3], Q[8]),
                (Q[8], Q[13]),
                (Q[0], Q[4]),
                (Q[4], Q[9]),
                (Q[9], Q[14]),
                (Q[14], Q[18]),
                (Q[1], Q[5]),
                (Q[5], Q[10]),
                (Q[10], Q[15]),
                (Q[15], Q[19]),
                (Q[2], Q[6]),
                (Q[6], Q[11]),
                (Q[11], Q[16]),
                (Q[16], Q[20]),
                (Q[7], Q[12]),
                (Q[12], Q[17]),
            ]
            chip.add_edges_from(graph_list_h + graph_list_v)
        elif connectivity == "5_qubits":
            Q = sympy.symbols([f"q{i}" for i in range(5)])
            chip = nx.Graph()
            chip.add_nodes_from(Q)
            graph_list = [
                (Q[0], Q[2]),
                (Q[1], Q[2]),
                (Q[3], Q[2]),
                (Q[4], Q[2]),
            ]
            chip.add_edges_from(graph_list)
        else:
            print("No connectivity map named %s found, 21 qubit chip will be used instead." % connectivity)
            return self.set_special_connectivity("21_qubits")
        return chip
