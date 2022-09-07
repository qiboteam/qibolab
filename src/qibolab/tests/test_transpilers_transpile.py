# -*- coding: utf-8 -*-
import itertools

import numpy as np
import pytest
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.models import Circuit

from qibolab.transpilers import transpile


def generate_random_circuit(nqubits, ngates, seed=None):
    """Generate random circuits one-qubit rotations and CZ gates."""
    pairs = list(itertools.combinations(range(nqubits), 2))
    if seed is not None:  # pragma: no cover
        np.random.seed(seed)

    one_qubit_gates = [gates.RX, gates.RY, gates.RZ, gates.X, gates.Y, gates.Z, gates.H]
    two_qubit_gates = [
        gates.CRX,
        gates.CRY,
        gates.CRZ,
        gates.CNOT,
        gates.CZ,
        gates.SWAP,
    ]
    n1, n2 = len(one_qubit_gates), len(two_qubit_gates)
    circuit = Circuit(nqubits)
    for _ in range(ngates):
        igate = int(np.random.randint(0, n1 + n2))
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


@pytest.mark.parametrize("nqubits", [3])
@pytest.mark.parametrize("ngates", [20, 50, 100])
def test_transpile(nqubits, ngates):
    backend = NumpyBackend()
    circuit = generate_random_circuit(nqubits, ngates)
    transpiled_circuit, _ = transpile(circuit)
    final_state = backend.execute_circuit(transpiled_circuit)
    target_circuit = backend.execute_circuit(circuit)
    np.testing.assert_allclose(final_state, target_circuit)
