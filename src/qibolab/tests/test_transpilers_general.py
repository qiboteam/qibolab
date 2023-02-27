import networkx as nx
import numpy as np
import pytest
import sympy
from qibo import gates
from qibo.models import Circuit

from qibolab.transpilers.general_connectivity import Transpiler


def generate_random_circuit(nqubits, ngates):
    """Generate a random circuit with RX and CZ gates."""
    one_qubit_gates = [gates.RX]
    two_qubit_gates = [gates.CZ]
    n1, n2 = len(one_qubit_gates), len(two_qubit_gates)
    n = n1 + n2 if nqubits > 1 else n1
    circuit = Circuit(nqubits)
    for _ in range(ngates):
        igate = int(np.random.randint(0, n))
        if igate >= n1:
            q = tuple(np.random.randint(0, nqubits, 2))
            while q[0] == q[1]:
                q = tuple(np.random.randint(0, nqubits, 2))
            gate = two_qubit_gates[igate - n1]
        else:
            q = (np.random.randint(0, nqubits),)
            gate = one_qubit_gates[igate]
        if issubclass(gate, gates.ParametrizedGate):
            theta = 2 * np.pi * np.random.random()
            circuit.add(gate(*q, theta=theta, trainable=False))
        else:
            circuit.add(gate(*q))
    return circuit


def custom_cicuit():
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.CZ(1, 0))
    circuit.add(gates.CZ(2, 1))
    return circuit


def test_simple_circuit():
    transpiler = Transpiler(connectivity="21_qubits", init_method="greedy", init_samples=0)
    circ = custom_cicuit()
    transpiled_circuit, final_map, initial_map, added_swaps = transpiler.transpile(circ)
    np.testing.assert_allclose(added_swaps, 2)


@pytest.mark.parametrize("gates", [5, 20, 50])
@pytest.mark.parametrize("qubits", [5, 10, 21])
def test_random_circuit(gates, qubits):
    transpiler = Transpiler(connectivity="21_qubits", init_method="greedy", init_samples=20)
    circ = generate_random_circuit(nqubits=qubits, ngates=gates)
    transpiled_circuit, final_map, initial_map, added_swaps = transpiler.transpile(circ)


def test_5q():
    transpiler = Transpiler(connectivity="5_qubits", init_method="greedy", init_samples=20)
    circ = generate_random_circuit(5, 20)
    transpiled_circuit, final_map, initial_map, added_swaps = transpiler.transpile(circ)


def test_subgraph_init():
    transpiler = Transpiler(connectivity="5_qubits", init_method="subgraph", init_samples=20)
    circ = generate_random_circuit(5, 20)
    transpiled_circuit, final_map, initial_map, added_swaps = transpiler.transpile(circ)


def test_custom_mapping():
    transpiler = Transpiler(connectivity="5_qubits")
    transpiler.custom_qubit_mapping({"q0": 1, "q1": 2, "q2": 3, "q3": 4, "q4": 0})
    circ = generate_random_circuit(5, 20)
    transpiled_circuit, final_map, initial_map, added_swaps = transpiler.transpile(circ)


def test_custom_connectivity():
    transpiler = Transpiler(init_method="greedy", init_samples=20)
    circ = generate_random_circuit(5, 20)
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
    transpiler.set_connectivity(chip)
    transpiled_circuit, final_map, initial_map, added_swaps = transpiler.transpile(circ)
