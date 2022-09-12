# -*- coding: utf-8 -*-
import numpy as np
import pytest
from qibo import gates
from qibo.models import Circuit

from qibolab.backends import QibolabBackend
from qibolab.platforms.abstract import AbstractPlatform

platform_names = ["tii1q", "tii5q"]  # 'qili' , 'icarusq']


@pytest.mark.xfail
@pytest.mark.parametrize("platform_name", platform_names)
def test_backend_init(platform_name):
    from qibolab.platforms.multiqubit import MultiqubitPlatform

    backend = QibolabBackend(platform_name)
    if platform_name in platform_names:
        assert isinstance(backend.platform, MultiqubitPlatform)


@pytest.mark.xfail
@pytest.mark.parametrize("platform_name", platform_names)
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
@pytest.mark.parametrize("platform_name", platform_names)
def test_execute_circuit(platform_name):
    backend = QibolabBackend(platform_name)
    platform: AbstractPlatform = backend.platform
    nqubits = platform.nqubits

    def generate_circuit_with_gate(gate, *params, **kwargs):
        _circuit = Circuit(nqubits)
        for qubit in range(nqubits):
            _circuit.add(gate(qubit, *params, **kwargs))
        qubits = [qubit for qubit in range(nqubits)]
        _circuit.add(gates.M(*qubits))
        return _circuit

    circuit = generate_circuit_with_gate(gates.I)
    result = backend.execute_circuit(circuit, nshots=100)

    circuit = generate_circuit_with_gate(gates.X)
    result = backend.execute_circuit(circuit, nshots=100)

    circuit = generate_circuit_with_gate(gates.Y)
    result = backend.execute_circuit(circuit, nshots=100)

    circuit = generate_circuit_with_gate(gates.Z)
    result = backend.execute_circuit(circuit, nshots=100)

    circuit = generate_circuit_with_gate(gates.RX, np.pi / 8)
    result = backend.execute_circuit(circuit, nshots=100)

    circuit = generate_circuit_with_gate(gates.RY, -np.pi / 8)
    result = backend.execute_circuit(circuit, nshots=100)

    circuit = generate_circuit_with_gate(gates.RZ, np.pi / 4)
    result = backend.execute_circuit(circuit, nshots=100)

    circuit = generate_circuit_with_gate(gates.U3, theta=0.1, phi=0.2, lam=0.3)
    result = backend.execute_circuit(circuit, nshots=100)


# TODO: speed up by instantiating the backend once per platform
# TODO: test other platforms (qili, icarusq)
# TODO: test_apply_gate
# TODO: test_apply_gate_density_matrix
# TODO: test_circuit_result_tensor
# TODO: test_circuit_result_representation
# TODO: test_circuit_result_probabilities
