"""Testing basic circuits with emulator platforms."""

import numpy as np
import pytest
from qibo import Circuit, construct_backend, gates


def test_measurement(platform):
    backend = construct_backend(backend="qibolab", platform=platform)
    circuit = Circuit(1)
    circuit.add(gates.M(0))
    result = backend.execute_circuit(circuit, nshots=1000)
    pytest.approx(result.samples().mean(), abs=5e-2) == 0


def test_hadamard(platform):
    backend = construct_backend(backend="qibolab", platform=platform)
    circuit = Circuit(1)
    circuit.add(gates.GPI2(0, 0))
    circuit.add(gates.M(0))
    result = backend.execute_circuit(circuit, nshots=1000)
    pytest.approx(result.samples().mean(), abs=1e-1) == 0.5


def test_rz(platform):
    backend = construct_backend(backend="qibolab", platform=platform)
    circuit = Circuit(1)
    circuit.add(gates.GPI2(0, 0))
    circuit.add(gates.RZ(0, np.pi / 2))
    circuit.add(gates.GPI2(0, np.pi / 2))
    circuit.add(gates.M(0))
    result = backend.execute_circuit(circuit, nshots=1000)
    pytest.approx(result.samples().mean(), abs=5e-2) == 1
