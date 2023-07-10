import itertools

import networkx as nx
import numpy as np
import pytest
from qibo import gates
from qibo.models import Circuit

from qibolab.native import NativeType
from qibolab.transpilers.pipeline import Passes, assert_transpiling


def generate_random_circuit(nqubits, ngates, seed=None):
    """Generate random circuits one-qubit rotations and CZ gates."""
    pairs = list(itertools.combinations(range(nqubits), 2))
    if seed is not None:  # pragma: no cover
        np.random.seed(seed)

    one_qubit_gates = [gates.RX, gates.RY, gates.RZ, gates.X, gates.Y, gates.Z, gates.H]
    two_qubit_gates = [
        gates.CNOT,
        gates.CZ,
        gates.SWAP,
        gates.iSWAP,
        gates.CRX,
        gates.CRY,
        gates.CRZ,
    ]
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
            circuit.add(gate(*q, theta=theta))
        else:
            circuit.add(gate(*q))
    return circuit


def star_connectivity():
    Q = ["q" + str(i) for i in range(5)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)
    graph_list = [(Q[i], Q[2]) for i in range(5) if i != 2]
    chip.add_edges_from(graph_list)
    return chip


@pytest.mark.parametrize("ngates", [5, 10, 50])
def test_pipeline_default(ngates):
    circ = generate_random_circuit(nqubits=5, ngates=ngates)
    default_transpiler = Passes(connectivity=star_connectivity())
    transpiled_circ, final_layout = default_transpiler(circ)
    print(circ.draw())
    print(transpiled_circ.draw())
    initial_layout = default_transpiler.get_initial_layout()
    assert_transpiling(
        circuit=transpiled_circ,
        connectivity=star_connectivity(),
        initial_layout=initial_layout,
        final_layout=final_layout,
        native_gates=NativeType.CZ,
    )
