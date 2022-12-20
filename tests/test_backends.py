import warnings

import numpy as np
import pytest
from qibo import gates
from qibo.models import Circuit

from qibolab.backends import QibolabBackend
from qibolab.platforms.abstract import AbstractPlatform


@pytest.fixture(scope="module")
def backend(request):
    backend = QibolabBackend(request.param)
    backend.platform.connect()
    backend.platform.setup()
    yield backend
    backend.platform.disconnect()


def generate_circuit_with_gate(nqubits, gate, **kwargs):
    circuit = Circuit(nqubits)
    circuit.add(gate(qubit, **kwargs) for qubit in range(nqubits))
    circuit.add(gates.M(*range(nqubits)))
    return circuit


@pytest.mark.qpu
def test_execute_circuit_errors(backend):
    circuit = Circuit(1)
    circuit.add(gates.X(0))
    circuit.add(gates.M(0))
    with pytest.raises(ValueError):
        backend.execute_circuit(circuit, initial_state=np.ones(2))


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
def test_execute_circuit(backend, gate, kwargs):
    nqubits = backend.platform.nqubits
    circuit = generate_circuit_with_gate(nqubits, gate, **kwargs)
    result = backend.execute_circuit(circuit, nshots=100)


@pytest.mark.qpu
def test_measurement_samples(backend):
    nqubits = backend.platform.nqubits
    circuit = Circuit(nqubits)
    circuit.add(gates.M(*range(nqubits)))
    result = backend.execute_circuit(circuit, nshots=100)
    assert result.samples().shape == (100, nqubits)
    assert sum(result.frequencies().values()) == 100


@pytest.mark.qpu
@pytest.mark.xfail(raises=AssertionError, reason="Probabilities are not well calibrated")
def test_ground_state_probabilities_circuit(backend):
    nqubits = backend.platform.nqubits
    circuit = Circuit(nqubits)
    circuit.add(gates.M(*range(nqubits)))
    result = backend.execute_circuit(circuit, nshots=5000)
    probs = result.probabilities()
    warnings.warn(f"Ground state probabilities: {probs}")
    np.testing.assert_allclose(probs, [1, 0], atol=0.05)


@pytest.mark.qpu
@pytest.mark.xfail(raises=AssertionError, reason="Probabilities are not well calibrated")
def test_excited_state_probabilities_circuit(backend):
    nqubits = backend.platform.nqubits
    circuit = Circuit(nqubits)
    circuit.add(gates.X(q) for q in range(nqubits))
    circuit.add(gates.M(*range(nqubits)))
    result = backend.execute_circuit(circuit, nshots=5000)
    probs = result.probabilities()
    warnings.warn(f"Excited state probabilities: {probs}")
    np.testing.assert_allclose(probs, [0, 1], atol=0.05)


# TODO: test other platforms (qili, icarusq)
# TODO: test_circuit_result_tensor
# TODO: test_circuit_result_representation
