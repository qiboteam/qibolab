import networkx as nx
import numpy as np
import pytest
from qibo import gates
from qibo.models import Circuit

from qibolab.transpilers.general_connectivity import Transpiler


def generate_random_circuit(nqubits, ngates, seed=42):
    """Generate a random circuit with RX and CZ gates."""
    np.random.seed(seed)
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


def custom_circuit():
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.CZ(1, 0))
    circuit.add(gates.CZ(2, 1))
    return circuit


def special_connectivity(connectivity):
    """Return a TII harware chip connectivity as a networkx graph"""
    if connectivity == "21_qubits":
        Q = [f"q{i}" for i in range(21)]
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
        Q = [f"q{i}" for i in range(5)]
        chip = nx.Graph()
        chip.add_nodes_from(Q)
        graph_list = [
            (Q[0], Q[2]),
            (Q[1], Q[2]),
            (Q[3], Q[2]),
            (Q[4], Q[2]),
        ]
        chip.add_edges_from(graph_list)
    return chip


def test_connectivity_and_samples():
    transpiler = Transpiler(connectivity=special_connectivity("21_qubits"), init_method="greedy", init_samples=20)
    assert transpiler.connectivity.number_of_nodes() == 21
    assert transpiler.init_samples == 20


def test_connectivity_setter_error():
    transpiler = Transpiler(connectivity=special_connectivity("21_qubits"), init_method="greedy", init_samples=20)
    with pytest.raises(TypeError):
        transpiler.connectivity = 1


def test_3q_error():
    circ = Circuit(3)
    circ.add(gates.TOFFOLI(0, 1, 2))
    transpiler = Transpiler(connectivity=special_connectivity("21_qubits"), init_method="greedy", init_samples=20)
    with pytest.raises(ValueError):
        transpiler.transpile(circ)


def test_insufficient_qubits():
    circ = generate_random_circuit(10, 10)
    transpiler = Transpiler(connectivity=special_connectivity("5_qubits"), init_method="greedy", init_samples=20)
    with pytest.raises(ValueError):
        transpiler.transpile(circ)


def test_simple_circuit():
    transpiler = Transpiler(connectivity=special_connectivity("21_qubits"), init_method="greedy", init_samples=0)
    circ = custom_circuit()
    transpiled_circuit, final_map, initial_map, added_swaps = transpiler.transpile(circ)
    np.testing.assert_allclose(added_swaps, 2)


@pytest.mark.parametrize("gates", [5, 20, 50])
@pytest.mark.parametrize("qubits", [5, 10, 21])
def test_random_circuit(gates, qubits):
    transpiler = Transpiler(connectivity=special_connectivity("21_qubits"), init_method="greedy", init_samples=20)
    circ = generate_random_circuit(nqubits=qubits, ngates=gates)
    transpiled_circuit, final_map, initial_map, added_swaps = transpiler.transpile(circ)


@pytest.mark.parametrize("gates", [20, 50])
def test_subgraph_init(gates):
    transpiler = Transpiler(connectivity=special_connectivity("5_qubits"), init_method="subgraph")
    circ = generate_random_circuit(5, gates)
    transpiled_circuit, final_map, initial_map, added_swaps = transpiler.transpile(circ)


def test_subgraph_init_simple():
    transpiler = Transpiler(connectivity=special_connectivity("5_qubits"), init_method="subgraph")
    circ = Circuit(3)
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(2, 1))
    transpiled_circuit, final_map, initial_map, added_swaps = transpiler.transpile(circ)
    assert added_swaps == 0


def test_custom_mapping():
    transpiler = Transpiler(connectivity=special_connectivity("5_qubits"))
    transpiler.custom_qubit_mapping([1, 2, 3, 4, 0])
    circ = generate_random_circuit(5, 20)
    transpiled_circuit, final_map, initial_map, added_swaps = transpiler.transpile(circ)


def test_custom_connectivity():
    transpiler = Transpiler(connectivity=special_connectivity("5_qubits"), init_method="greedy", init_samples=20)
    circ = generate_random_circuit(5, 20)
    Q = [f"q{i}" for i in range(5)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)
    graph_list = [
        (Q[0], Q[2]),
        (Q[1], Q[2]),
        (Q[3], Q[2]),
        (Q[4], Q[2]),
    ]
    chip.add_edges_from(graph_list)
    transpiler.connectivity = chip
    transpiled_circuit, final_map, initial_map, added_swaps = transpiler.transpile(circ)
