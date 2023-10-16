import networkx as nx
import numpy as np
import pytest
from qibo import gates
from qibo.models import Circuit

from qibolab.transpilers.placer import (
    Custom,
    PlacementError,
    Subgraph,
    Trivial,
    assert_placement,
)
from qibolab.transpilers.router import (
    CircuitMap,
    ConnectivityError,
    ShortestPaths,
    assert_connectivity,
    remap_circuit,
)


def star_connectivity():
    Q = ["q" + str(i) for i in range(5)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)
    graph_list = [(Q[i], Q[2]) for i in range(5) if i != 2]
    chip.add_edges_from(graph_list)
    return chip


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


def matched_circuit():
    """Return a simple circuit that can be executed on star connectivity"""
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.Z(1))
    circuit.add(gates.CZ(2, 1))
    circuit.add(gates.M(0))
    return circuit


def test_assert_connectivity():
    assert_connectivity(star_connectivity(), matched_circuit())


def test_assert_connectivity_false():
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 1))
    with pytest.raises(ConnectivityError):
        assert_connectivity(star_connectivity(), circuit)


def test_assert_connectivity_3q():
    circuit = Circuit(5)
    circuit.add(gates.TOFFOLI(0, 1, 2))
    with pytest.raises(ConnectivityError):
        assert_connectivity(star_connectivity(), circuit)


def test_remap_circuit():
    circuit = Circuit(3)
    circuit.add(gates.CNOT(1, 0))
    circuit.add(gates.CNOT(2, 0))
    circuit.add(gates.CNOT(1, 2))
    qubit_map = np.asarray([2, 1, 0])
    ref_circuit = Circuit(3)
    ref_circuit.add(gates.CNOT(1, 2))
    ref_circuit.add(gates.CNOT(0, 2))
    ref_circuit.add(gates.CNOT(1, 0))
    new_circuit = remap_circuit(circuit, qubit_map)
    for i, gate in enumerate(new_circuit.queue):
        assert gate.qubits == ref_circuit.queue[i].qubits


@pytest.mark.parametrize("split", [2.0, -1.0])
def test_split_setter(split):
    with pytest.raises(ValueError):
        transpiler = ShortestPaths(connectivity=star_connectivity(), sampling_split=split)


def test_insufficient_qubits():
    circuit = generate_random_circuit(10, 20)
    placer = Trivial()
    initial_layout = placer(circuit)
    transpiler = ShortestPaths(connectivity=star_connectivity())
    with pytest.raises(ValueError):
        transpiler(circuit, initial_layout)


def test_incorrect_initial_layout():
    placer = Trivial()
    circuit = Circuit(4)
    circuit.add(gates.CNOT(1, 0))
    circuit.add(gates.CNOT(2, 0))
    circuit.add(gates.CNOT(3, 0))
    initial_layout = placer(circuit)
    transpiler = ShortestPaths(connectivity=star_connectivity())
    with pytest.raises(PlacementError):
        transpiler(circuit, initial_layout)


@pytest.mark.parametrize("gates", [1, 10, 50])
@pytest.mark.parametrize("qubits", [3, 5])
def test_random_circuits_5q(gates, qubits):
    placer = Trivial()
    layout_circ = Circuit(5)
    initial_layout = placer(layout_circ)
    transpiler = ShortestPaths(connectivity=star_connectivity(), verbose=True)
    circuit = generate_random_circuit(nqubits=qubits, ngates=gates)
    transpiled_circuit, final_qubit_map = transpiler(circuit, initial_layout)
    assert transpiler.added_swaps >= 0
    assert_connectivity(star_connectivity(), transpiled_circuit)
    assert_placement(transpiled_circuit, final_qubit_map)
    assert gates + transpiler.added_swaps == transpiled_circuit.ngates


def q21_connectivity():
    """Returns connectivity map for the TII 21 qubit chip"""
    Q = ["q" + str(i) for i in range(21)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)
    graph_list_h = [(Q[i], Q[i + 1]) for i in range(20) if i % 5 != 2]
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


@pytest.mark.parametrize("gates", [5, 30])
@pytest.mark.parametrize("qubits", [10, 21])
@pytest.mark.parametrize("split", [1.0, 0.5, 0.1])
def test_random_circuits_21q(gates, qubits, split):
    placer = Trivial()
    layout_circ = Circuit(21)
    initial_layout = placer(layout_circ)
    transpiler = ShortestPaths(connectivity=q21_connectivity(), sampling_split=split)
    circuit = generate_random_circuit(nqubits=qubits, ngates=gates)
    transpiled_circuit, final_qubit_map = transpiler(circuit, initial_layout)
    assert transpiler.added_swaps >= 0
    assert_connectivity(q21_connectivity(), transpiled_circuit)
    assert_placement(transpiled_circuit, final_qubit_map)
    assert gates + transpiler.added_swaps == transpiled_circuit.ngates


def star_circuit():
    circuit = Circuit(5)
    for i in range(1, 5):
        circuit.add(gates.CNOT(i, 0))
    return circuit


def test_star_circuit():
    placer = Subgraph(star_connectivity())
    initial_layout = placer(star_circuit())
    transpiler = ShortestPaths(connectivity=star_connectivity())
    transpiled_circuit, final_qubit_map = transpiler(star_circuit(), initial_layout)
    assert transpiler.added_swaps == 0
    assert_connectivity(star_connectivity(), transpiled_circuit)
    assert_placement(transpiled_circuit, final_qubit_map)
    assert final_qubit_map["q2"] == 0


def test_star_circuit_custom_map():
    placer = Custom(map=[1, 0, 2, 3, 4], connectivity=star_connectivity())
    initial_layout = placer()
    transpiler = ShortestPaths(connectivity=star_connectivity())
    transpiled_circuit, final_qubit_map = transpiler(star_circuit(), initial_layout)
    assert transpiler.added_swaps == 1
    assert_connectivity(star_connectivity(), transpiled_circuit)
    assert_placement(transpiled_circuit, final_qubit_map)
    assert final_qubit_map == {"q0": 1, "q1": 2, "q2": 0, "q3": 3, "q4": 4}


def test_routing_with_measurements():
    placer = Trivial(connectivity=star_connectivity())
    circuit = Circuit(5)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.M(0, 2, 3))
    initial_layout = placer(circuit=circuit)
    transpiler = ShortestPaths(connectivity=star_connectivity())
    transpiled_circuit, _ = transpiler(circuit, initial_layout)
    print(transpiled_circuit.draw())
    print(circuit.draw())
    assert transpiled_circuit.ngates == 3
    measured_qubits = transpiled_circuit.queue[2].qubits
    assert measured_qubits == (0, 1, 3)


def test_circuit_map():
    circ = Circuit(4)
    circ.add(gates.H(1))
    circ.add(gates.H(0))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.H(0))
    circ.add(gates.CZ(1, 2))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(2, 3))
    initial_layout = {"q0": 2, "q1": 0, "q2": 1, "q3": 3}
    circuit_map = CircuitMap(initial_layout=initial_layout, circuit=circ)
    block_list = circuit_map.circuit_blocks
    # test blocks_qubits_pairs
    assert circuit_map.blocks_qubits_pairs() == [(0, 1), (1, 2), (0, 1), (2, 3)]
    # test execute_block and routed_circuit
    circuit_map.execute_block(block_list.search_by_index(0))
    routed_circuit = circuit_map.routed_circuit()
    assert isinstance(routed_circuit.queue[0], gates.H)
    assert len(routed_circuit.queue) == 4
    assert routed_circuit.queue[2].qubits == (1, 2)
    # test update
    circuit_map.update((0, 2))
    routed_circuit = circuit_map.routed_circuit()
    assert isinstance(routed_circuit.queue[4], gates.SWAP)
    assert routed_circuit.queue[4].qubits == (1, 0)
    assert circuit_map._swaps == 1
    assert circuit_map._circuit_logical == [2, 1, 0, 3]
    circuit_map.update((1, 2))
    routed_circuit = circuit_map.routed_circuit()
    assert routed_circuit.queue[5].qubits == (2, 0)
    assert circuit_map._circuit_logical == [1, 2, 0, 3]
    # test execute_block after multiple swaps
    circuit_map.execute_block(block_list.search_by_index(1))
    circuit_map.execute_block(block_list.search_by_index(2))
    circuit_map.execute_block(block_list.search_by_index(3))
    routed_circuit = circuit_map.routed_circuit()
    assert isinstance(routed_circuit.queue[6], gates.CZ)
    # circuit to logical map: [1,2,0,3]. initial map: {"q0": 2, "q1": 0, "q2": 1, "q3": 3}.
    assert routed_circuit.queue[6].qubits == (0, 1)  # initial circuit qubits (1,2)
    assert routed_circuit.queue[7].qubits == (2, 0)  # (0,1)
    assert routed_circuit.queue[8].qubits == (1, 3)  # (2,3)
    assert len(circuit_map.circuit_blocks()) == 0
