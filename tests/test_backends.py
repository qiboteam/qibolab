import warnings

import numpy as np
import pytest
from qibo import gates
from qibo.models import Circuit

from qibolab.backends import QibolabBackend


def generate_circuit_with_gate(nqubits, gate, **kwargs):
    circuit = Circuit(nqubits)
    circuit.add(gate(qubit, **kwargs) for qubit in range(nqubits))
    circuit.add(gates.M(*range(nqubits)))
    return circuit


@pytest.fixture(scope="module")
def connected_backend(connected_platform):
    connected_platform.setup()
    connected_platform.start()
    yield QibolabBackend(connected_platform)
    connected_platform.stop()


def test_execute_circuit_initial_state():
    backend = QibolabBackend("dummy")
    circuit = Circuit(1)
    circuit.add(gates.X(0))
    circuit.add(gates.M(0))
    with pytest.raises(ValueError):
        backend.execute_circuit(circuit, initial_state=np.ones(2))

    initial_circuit = Circuit(1)
    initial_circuit.add(gates.H(0))
    backend.execute_circuit(circuit, initial_state=initial_circuit)


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
def test_execute_circuit(gate, kwargs):
    backend = QibolabBackend("dummy")
    nqubits = backend.platform.nqubits
    circuit = generate_circuit_with_gate(nqubits, gate, **kwargs)
    result = backend.execute_circuit(circuit, nshots=100)


def test_measurement_samples():
    backend = QibolabBackend("dummy")
    nqubits = backend.platform.nqubits

    circuit = Circuit(nqubits)
    circuit.add(gates.M(*range(nqubits)))
    result = backend.execute_circuit(circuit, nshots=100)
    assert result.samples().shape == (100, nqubits)
    assert sum(result.frequencies().values()) == 100

    circuit = Circuit(nqubits)
    circuit.add(gates.M(0, 2))
    result = backend.execute_circuit(circuit, nshots=100)
    assert result.samples().shape == (100, 2)
    assert sum(result.frequencies().values()) == 100


def test_execute_circuits():
    backend = QibolabBackend("dummy")
    circuit = Circuit(3)
    circuit.add(gates.H(i) for i in range(3))
    circuit.add(gates.M(0, 1, 2))

    results = backend.execute_circuits(5 * [circuit], nshots=100)
    assert len(results) == 5
    for result in results:
        assert result.samples().shape == (100, 3)
        assert sum(result.frequencies().values()) == 100


@pytest.mark.qpu
@pytest.mark.xfail(raises=AssertionError, reason="Probabilities are not well calibrated")
def test_ground_state_probabilities_circuit(connected_backend):
    nshots = 5000
    nqubits = connected_backend.platform.nqubits
    circuit = Circuit(nqubits)
    circuit.add(gates.M(*range(nqubits)))
    result = connected_backend.execute_circuit(circuit, nshots=nshots)
    freqs = result.frequencies(binary=False)
    probs = [freqs[i] / nshots for i in range(2**nqubits)]
    warnings.warn(f"Ground state probabilities: {probs}")
    target_probs = np.zeros(2**nqubits)
    target_probs[0] = 1
    np.testing.assert_allclose(probs, target_probs, atol=0.05)


@pytest.mark.qpu
@pytest.mark.xfail(raises=AssertionError, reason="Probabilities are not well calibrated")
def test_excited_state_probabilities_circuit(connected_backend):
    nshots = 5000
    nqubits = connected_backend.platform.nqubits
    circuit = Circuit(nqubits)
    circuit.add(gates.X(q) for q in range(nqubits))
    circuit.add(gates.M(*range(nqubits)))
    result = connected_backend.execute_circuit(circuit, nshots=nshots)
    freqs = result.frequencies(binary=False)
    probs = [freqs[i] / nshots for i in range(2**nqubits)]
    warnings.warn(f"Excited state probabilities: {probs}")
    target_probs = np.zeros(2**nqubits)
    target_probs[-1] = 1
    np.testing.assert_allclose(probs, target_probs, atol=0.05)


@pytest.mark.qpu
@pytest.mark.xfail(raises=AssertionError, reason="Probabilities are not well calibrated")
def test_superposition_for_all_qubits(connected_backend):
    """Applies an H gate to each qubit of the circuit and measures the probabilities."""
    nshots = 5000
    nqubits = connected_backend.platform.nqubits
    probs = []
    for q in range(nqubits):
        circuit = Circuit(nqubits)
        circuit.add(gates.H(q=q))
        circuit.add(gates.M(q))
        freqs = connected_backend.execute_circuit(circuit, nshots=nshots).frequencies(binary=False)
        probs.append([freqs[i] / nshots for i in range(2)])
        warnings.warn(f"Probabilities after an Hadamard gate applied to qubit {q}: {probs[-1]}")
    probs = np.asarray(probs)
    target_probs = np.repeat(a=0.5, repeats=nqubits)
    np.testing.assert_allclose(probs.T[0], target_probs, atol=0.05)
    np.testing.assert_allclose(probs.T[1], target_probs, atol=0.05)


# TODO: test_circuit_result_tensor
# TODO: test_circuit_result_representation
