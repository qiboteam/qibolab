import random
from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import sympy
from qibo import gates
from qibo.models import Circuit


class Transpiler:
    def __init__(self, connectivity="21_qubits", init_method="greedy", init_samples=100):
        self.connectivity = self.set_connectivity(connectivity)
        self.init_method = init_method
        self.init_samples = init_samples

    def set_connectivity(self, connectivity):
        if connectivity == "21_qubits":
            return self.set_special_connectivity()
        else:
            # TODO check that it is a real connectivity map
            return connectivity

    def draw_connectivity(self):
        """Show connectivity graph"""
        pos = nx.spectral_layout(self.connectivity)
        nx.draw(self.connectivity, pos=pos, with_labels=True)
        plt.show()

    def set_init_method(self, init_method, init_samples=100):
        """Set the initial mapping for the transpiler."""
        self.init_method = init_method
        self.init_samples = init_samples

    def get_connectivity(self):
        return self.connectivity

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

    def greedy_init(self):
        """initialize the circuit with greedy algorithm let a maximum number of 2-qubit gates can be applied without introducing any SWAP gate"""
        nodes = self.connectivity.number_of_nodes()
        keys = list(self.connectivity.nodes())
        values = [i for i in range(nodes)]
        final_mapping = {keys[i]: values[i] for i in range(len(keys))}
        final_graph = nx.relabel_nodes(self.connectivity, final_mapping)
        final_cost = len(self.reduce(final_graph))
        for i in range(self.init_samples):
            random.shuffle(values)
            mapping = {keys[i]: values[i] for i in range(len(keys))}
            graph = nx.relabel_nodes(self.connectivity, mapping)
            cost = len(self.reduce(graph))
            if cost == 0:
                return final_graph, final_mapping
            if cost < final_cost:
                final_graph = graph
                final_mapping = mapping
                final_cost = cost
        return final_graph, final_mapping

    def map_list(self, path):
        """Return all possible walks of qubits for a given path"""
        keys = path
        path_ends = [path[0]] + [path[-1]]
        path_middle = path[1:-1]
        List = []
        for i in range(len(path) - 1):
            values = path_middle[:i] + path_ends + path_middle[i:]
            mapping = {keys[i]: values[i] for i in range(len(keys))}
            List.append(mapping)
        return List

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
        # Reduce the number of paths to be faster
        for i in range(len(path_list)):
            path = path_list[i]
            List = self.map_list(path)
            for j in range(len(List)):
                mapping = List[j]
                new_graph = nx.relabel_nodes(self.graph, mapping)
                new_circuit = self.reduce(new_graph)
                # Greedy looking for the optimal path and the optimal walk on this path
                if len(new_circuit) < len(final_circuit):
                    final_graph = new_graph
                    final_circuit = new_circuit
                    final_mapping = mapping
        return final_graph, final_circuit, final_mapping

    def transpile(self, qibo_circuit):
        self.circuit_repr = self.translate_circuit(qibo_circuit)
        self.mapping_list = []
        print("Initial Circuit: ", self.circuit_repr)
        self.circuit_repr = self.clean()
        print("Cleaned Circuit: ", self.circuit_repr)
        if self.init_method == "greedy":
            self.graph, self.mapping = self.greedy_init()
        else:
            print("ERROR")
        print("Graph: ", self.graph)
        print("Mapping: ", self.mapping)
        self.mapping_list.append(self.mapping)
        self.circuit_repr = self.reduce(self.graph)
        print("Circuit after first reduce:")
        print(self.circuit_repr)
        self.n_swap = 0
        while len(self.circuit_repr) != 0:
            self.graph, self.circuit_repr, self.mapping = self.relocate()
            self.mapping_list.append(self.mapping)
            print("STEP")
            print("Mapping: ", self.mapping)
            print("Circuit: ", self.circuit_repr)
            print("Added swaps: ", self.n_swap)

    def set_special_connectivity(self):
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
        return chip
