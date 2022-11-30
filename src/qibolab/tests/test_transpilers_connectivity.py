import itertools

import numpy as np
import pytest
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.models import Circuit

from qibolab.transpilers.connectivity import fix_connectivity, respects_connectivity


def generate_random_circuit(nqubits, depth, seed=None):
    """Generate random circuits one-qubit rotations and CZ gates."""
    pairs = list(itertools.combinations(range(nqubits), 2))
    if seed is not None:  # pragma: no cover
        np.random.seed(seed)

    rotations = [gates.RX, gates.RY, gates.RZ]
    circuit = Circuit(nqubits)
    for _ in range(depth):
        for i in range(nqubits):
            # generate a random rotation
            rotation = rotations[int(np.random.randint(0, 3))]
            theta = 2 * np.pi * np.random.random()
            circuit.add(rotation(i, theta=theta))
        # add CZ gates on random qubit pairs
        for i in np.random.randint(0, len(pairs), len(pairs)):
            q1, q2 = pairs[i]
            circuit.add(gates.CZ(q1, q2))

    return circuit


def transpose_qubits(state, qubits):
    """Reorders qubits of a given state vector."""
    original_shape = state.shape
    state = np.reshape(state, len(qubits) * (2,))
    state = np.transpose(state, qubits)
    return np.reshape(state, original_shape)


@pytest.mark.parametrize("run_number", range(5))
@pytest.mark.parametrize("nqubits", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("depth", [2, 5, 8])
def test_fix_connectivity(run_number, nqubits, depth):
    """Checks that the transpiled circuit can be executed and is equivalent to original."""
    original = generate_random_circuit(nqubits, depth)
    transpiled, hardware_qubits = fix_connectivity(original)
    # check that transpiled circuit can be executed
    assert respects_connectivity(transpiled)
    # check that execution results agree with original (using simulation)
    backend = NumpyBackend()
    final_state = backend.execute_circuit(transpiled).state()
    target_state = backend.execute_circuit(original).state()
    target_state = transpose_qubits(target_state, hardware_qubits)
    np.testing.assert_allclose(final_state, target_state)
