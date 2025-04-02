"""Testing basic circuits with emulator platforms."""

import numpy as np
from qibo import Circuit, construct_backend, gates


def test_measurement(platform):
    backend = construct_backend(backend="qibolab", platform=platform)
    circuit = Circuit(1)
    circuit.add(gates.M(0))
    result = backend.execute_circuit(circuit, nshots=1000)
    np.testing.assert_allclose(result.probabilities(), [1, 0], atol=5e-2)


def test_hadamard(platform):
    backend = construct_backend(backend="qibolab", platform=platform)
    circuit = Circuit(1)
    circuit.add(gates.GPI2(0, 0))
    circuit.add(gates.M(0))
    result = backend.execute_circuit(circuit, nshots=1000)
    np.testing.assert_allclose(result.probabilities(), [0.5, 0.5], atol=5e-2)


def test_rz(platform):
    backend = construct_backend(backend="qibolab", platform=platform)
    circuit = Circuit(1)
    circuit.add(gates.GPI2(0, 0))
    circuit.add(gates.RZ(0, np.pi / 2))
    circuit.add(gates.GPI2(0, np.pi / 2))
    circuit.add(gates.M(0))
    result = backend.execute_circuit(circuit, nshots=1000)
    np.testing.assert_allclose(result.probabilities(), [0, 1], atol=5e-2)
