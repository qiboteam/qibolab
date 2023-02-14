import random
from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import sympy
from qibo import gates
from qibo.models import Circuit


class Transpiler:
    def __init__(self, connectivity="21_qubits", init_method="greedy"):
        self.connectivity = self.set_connectivity(connectivity)
        self.init_method = init_method

    def set_connectivity(self, connectivity):
        if connectivity == "21_qubits":
            return self.set_special_connectivity()
        else:
            # TODO check that ie is a real connectivity map
            return connectivity

    def set_special_connectivity(self):
        Q = sympy.symbols([f"q{i+1}" for i in range(21)])
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

    def draw_connectivity(self):
        """Show connectivity graph"""
        pos = nx.spectral_layout(self.connectivity)
        nx.draw(self.connectivity, pos=pos, with_labels=True)
        plt.show()

    def set_init_method(self, init_method):
        self.init_method = init_method

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

    def reduce(self):
        """Reduce the circuit, delete a 2-qubit gate if it can be applied on the current configuration"""
        new_circuit = deepcopy(self.circuit_repr)
        while new_circuit != [] and (new_circuit[0][0], new_circuit[0][1]) in self.connectivity.edges():
            del new_circuit[0]
        return new_circuit

    def greedy_init(self):
        """initialize the circuit with greedy algorithm let a maximum number of 2-qubit gates can be applied without introducing any SWAP gate"""
        n = self.connectivity.number_of_nodes()
        keys = list(self.connectivity.nodes())
        values = [i for i in range(1, n + 1)]
        final_mapping = {keys[i]: values[i] for i in range(len(keys))}
        final_G = nx.relabel_nodes(self.connectivity, final_mapping)
        final_cost = len(self.reduce(final_G, self.circuit_repr))
        for i in range(self.connectivity):
            random.shuffle(values)
            mapping = {keys[i]: values[i] for i in range(len(keys))}
            G = nx.relabel_nodes(self.connectivity, mapping)
            cost = len(self.reduce(G, self.circuit_repr))
            if cost == 0:
                return final_G, final_mapping
            if cost < final_cost:
                final_G = G
                final_mapping = mapping
                final_cost = cost
        return final_G, final_mapping

    def transpile(self, qibo_circuit):
        self.circuit_repr = self.translate_circuit(qibo_circuit)
        print(self.circuit_repr)
        self.circuit_repr = self.clean()
        print(self.circuit_repr)
        if self.init_method == "greedy":
            self.greedy_init()
        else:
            print("ERROR")
