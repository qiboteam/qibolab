# -*- coding: utf-8 -*-
import numpy as np
import pytest
from qibo import gates
from qibo.models import Circuit

from qibolab.backends import QibolabBackend
from qibolab.platforms.abstract import AbstractPlatform


def generate_circuit_with_gate(nqubits, gate, *params, **kwargs):
    circuit = Circuit(nqubits)
    circuit.add(gate(q, *params, **kwargs) for q in range(nqubits))
    circuit.add(gates.M(*range(nqubits)))
    return circuit


@pytest.mark.qpu
def test_backend_init(platform_name):
    from qibolab.platforms.multiqubit import MultiqubitPlatform

    backend = QibolabBackend(platform_name)


@pytest.mark.qpu
def test_execute_circuit_errors(platform_name):
    backend = QibolabBackend(platform_name)
    circuit = Circuit(1)
    circuit.add(gates.X(0))
    circuit.add(gates.M(0))
    with pytest.raises(ValueError):
        result = backend.execute_circuit(circuit, initial_state=np.ones(2))


@pytest.mark.qpu
@pytest.mark.parametrize(
    "gateargs",
    [
        (gates.I,),
        (gates.X,),
        (gates.Y,),
        (gates.Z,),
        (gates.RX, np.pi / 8),
        (gates.RY, -np.pi / 8),
        (gates.RZ, np.pi / 4),
        (gates.U3, 0.1, 0.2, 0.3),
    ],
)
def test_execute_circuit(platform_name, gateargs):
    backend = QibolabBackend(platform_name)
    nqubits = backend.platform.nqubits
    circuit = generate_circuit_with_gate(nqubits, *gateargs)
    result = backend.execute_circuit(circuit, nshots=100)


@pytest.mark.qpu
def test_measurement_samples(platform_name):
    backend = QibolabBackend(platform_name)
    nqubits = backend.platform.nqubits
    circuit = Circuit(nqubits)
    circuit.add(gates.M(*range(nqubits)))
    result = backend.execute_circuit(circuit, nshots=100)
    assert result.samples().shape == (100, nqubits)
    assert sum(result.frequencies().values()) == 100


# TODO: speed up by instantiating the backend once per platform
# TODO: test other platforms (qili, icarusq)
# TODO: test_circuit_result_tensor
# TODO: test_circuit_result_representation
# TODO: test_circuit_result_probabilities
