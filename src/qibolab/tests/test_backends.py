# -*- coding: utf-8 -*-
import numpy as np
import pytest
from qibo import gates
from qibo.models import Circuit

from qibolab.backends import QibolabBackend
from qibolab.platforms.abstract import AbstractPlatform


def generate_circuit_with_gate(nqubits, gate, **kwargs):
    circuit = Circuit(nqubits)
    circuit.add(gate(qubit, **kwargs) for qubit in range(nqubits))
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
    with pytest.raises(RuntimeError):
        result = backend.execute_circuit(circuit)
    circuit.add(gates.M(0))
    with pytest.raises(ValueError):
        result = backend.execute_circuit(circuit, initial_state=np.ones(2))


@pytest.mark.qpu
@pytest.mark.parametrize(
    "gate,kwargs",
    [
        (gates.I, {}),
        (gates.X, {}),
        (gates.Y, {}),
        (gates.Z, {}),
        (gates.RX, {"theta": np.pi / 8}),
        (gates.RY, {"theta": -np.pi / 8}),
        (gates.RZ, {"theta": np.pi / 4}),
        (gates.U3, {"theta": 0.1, "phi": 0.2, "lam": 0.3}),
    ],
)
def test_execute_circuit(platform_name, gate, kwargs):
    backend = QibolabBackend(platform_name)
    platform: AbstractPlatform = backend.platform
    nqubits = platform.nqubits
    circuit = generate_circuit_with_gate(nqubits, gate, **kwargs)
    result = backend.execute_circuit(circuit, nshots=100)


# TODO: speed up by instantiating the backend once per platform
# TODO: test other platforms (qili, icarusq)
# TODO: test_apply_gate
# TODO: test_apply_gate_density_matrix
# TODO: test_circuit_result_tensor
# TODO: test_circuit_result_representation
# TODO: test_circuit_result_probabilities
