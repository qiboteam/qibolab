# -*- coding: utf-8 -*-
import numpy as np
import pytest
from qibo import gates
from qibo.models import Circuit

from qibolab.backends import QibolabBackend


@pytest.mark.parametrize("platform_name", ["tii1q", "tii5q"])
def test_backend_init(platform_name):
    from qibolab.platforms.multiqubit import MultiqubitPlatform

    backend = QibolabBackend(platform_name)
    if platform_name in ["tii1q", "tii5q"]:
        assert isinstance(backend.platform, MultiqubitPlatform)


@pytest.mark.parametrize("platform_name", ["tii1q", "tii5q"])
def test_execute_circuit_errors(platform_name):
    backend = QibolabBackend(platform_name)
    circuit = Circuit(1)
    circuit.add(gates.X(0))
    with pytest.raises(RuntimeError):
        result = backend.execute_circuit(circuit)
    circuit.add(gates.M(0))
    with pytest.raises(ValueError):
        result = backend.execute_circuit(circuit, initial_state=np.ones(2))


@pytest.mark.xfail
@pytest.mark.parametrize("platform_name", ["tii1q", "tii5q"])
def test_execute_circuit(platform_name):
    # TODO: Test this method on IcarusQ
    backend = QibolabBackend(platform_name)
    nqubits = backend.platform.nqubits
    circuit = Circuit(nqubits)
    for qubit in range(nqubits):
        circuit.add(gates.X(qubit))
    qubits = [qubit for qubit in range(nqubits)]
    circuit.add(gates.M(*qubits))
    result = backend.execute_circuit(circuit, nshots=100)


# TODO: test_circuit_result_representation
# TODO: test_circuit_result_probabilities
