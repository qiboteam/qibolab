import networkx as nx
import numpy as np
import pytest
from qibo import gates
from qibo.models import Circuit

from qibolab.transpilers.gate_decompositions import TwoQubitNatives
from qibolab.transpilers.routing import ShortestPaths


def star_connectivity():
    Q = ["q" + str(i) for i in range(5)]
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


def star_circuit():
    circuit = Circuit(5)
    circuit.add(gates.CNOT(1, 0))
    circuit.add(gates.CNOT(2, 0))
    circuit.add(gates.CNOT(3, 0))
    circuit.add(gates.CNOT(4, 0))
    return circuit


def generate_random_circuit(nqubits, ngates, seed=42):
    """Generate a random circuit with RX and CZ gates."""
    np.random.seed(seed)
    one_qubit_gates = [gates.RX, gates.RY, gates.RZ]
    two_qubit_gates = [gates.CZ, gates.CNOT, gates.SWAP]
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


def test_can_execute():
    transpiler = ShortestPaths(connectivity=star_connectivity(), verbose=True)
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.Z(1))
    circuit.add(gates.CZ(2, 1))
    circuit.add(gates.M(0))
    assert transpiler.is_satisfied(circuit)


def test_cannot_execute_connectivity():
    transpiler = ShortestPaths(connectivity=star_connectivity())
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 1))
    assert not transpiler.is_satisfied(circuit)


def test_cannot_execute_native_3q():
    transpiler = ShortestPaths(connectivity=star_connectivity())
    circuit = Circuit(5)
    circuit.add(gates.TOFFOLI(0, 1, 2))
    assert not transpiler.is_satisfied(circuit)


# def test_connectivity_and_samples():
#     transpiler = ShortestPaths(
#         connectivity=star_connectivity(), init_method="greedy", init_samples=20
#     )
#     assert transpiler.connectivity.number_of_nodes() == 21
#     assert transpiler.init_samples == 20


# def test_connectivity_setter_error():
#     transpiler = ShortestPaths(
#         connectivity=star_connectivity(), init_method="greedy", init_samples=20
#     )
#     with pytest.raises(TypeError):
#         transpiler.connectivity = 1


# def test_3q_error():
#     circ = Circuit(3)
#     circ.add(gates.TOFFOLI(0, 1, 2))
#     transpiler = ShortestPaths(
#         connectivity=star_connectivity(), init_method="greedy", init_samples=20
#     )
#     with pytest.raises(ValueError):
#         transpiler.transpile(circ)


# def test_insufficient_qubits():
#     circ = generate_random_circuit(10, 10)
#     transpiler = ShortestPaths(
#         connectivity=star_connectivity(), init_method="greedy", init_samples=20
#     )
#     with pytest.raises(ValueError):
#         transpiler.transpile(circ)


# @pytest.mark.parametrize("gates", [1, 10, 50])
# @pytest.mark.parametrize("qubits", [5, 21])
# @pytest.mark.parametrize(
#     "natives", [TwoQubitNatives.CZ, TwoQubitNatives.iSWAP, TwoQubitNatives.CZ | TwoQubitNatives.iSWAP]
# )
# def test_random_circuits(gates, qubits, natives):
#     transpiler = ShortestPaths(
#         connectivity=star_connectivity(), init_method="greedy", init_samples=50
#     )
#     circ = generate_random_circuit(nqubits=qubits, ngates=gates)
#     transpiled_circuit, qubit_map = transpiler.transpile(circ)
#     assert len(transpiler.initial_map) == 21 and len(transpiler.final_map) == 21
#     assert transpiler.added_swaps >= 0
#     assert transpiler.is_satisfied(transpiled_circuit)


# def test_split_setter():
#     with pytest.raises(ValueError):
#         transpiler = ShortestPaths(
#             connectivity=star_connectivity(), init_method="subgraph", sampling_split=2.0
#         )

# def test_split():
#     transpiler = ShortestPaths(
#         connectivity=star_connectivity(), init_method="greedy", init_samples=20, sampling_split=0.2
#     )
#     circ = generate_random_circuit(21, 50)
#     transpiled_circuit, qubit_map = transpiler.transpile(circ)
#     assert transpiler.added_swaps >= 0
#     assert len(transpiler.initial_map) == 21 and len(transpiler.final_map) == 21
#     assert transpiler.is_satisfied(transpiled_circuit)


# @pytest.mark.parametrize("one_q", [True, False])
# @pytest.mark.parametrize("two_q", [True, False])
# def test_fusion_algorithms(one_q, two_q):
#     transpiler = ShortestPaths(
#         connectivity=star_connectivity(), init_method="greedy", init_samples=20
#     )
#     circ = generate_random_circuit(21, 50)
#     transpiled_circuit, qubit_map = transpiler.transpile(circ)
#     assert transpiler.added_swaps >= 0
#     assert len(transpiler.initial_map) == 21 and len(transpiler.final_map) == 21
#     assert transpiler.is_satisfied(transpiled_circuit)
