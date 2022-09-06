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

    one_qubit = [gates.RX, gates.RY, gates.RZ, gates.X, gates.Y, gates.Z, gates.H]
    two_qubit = [gates.RX, gates.RY, gates.RZ]
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
