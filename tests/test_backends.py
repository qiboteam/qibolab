import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from qibo import gates
from qibo.models import Circuit

from qibolab import MetaBackend, create_platform
from qibolab.backends import QibolabBackend


def generate_circuit_with_gate(nqubits, gate, **kwargs):
    circuit = Circuit(nqubits)
    circuit.add(gate(qubit, **kwargs) for qubit in range(nqubits))
    circuit.add(gates.M(*range(nqubits)))
    return circuit


@pytest.fixture(scope="module")
def connected_backend(connected_platform):
    yield QibolabBackend(connected_platform)


def test_execute_circuit_initial_state():
    backend = QibolabBackend("dummy")
    circuit = Circuit(1)
    circuit.add(gates.GPI2(0, phi=0))
    circuit.add(gates.M(0))
    with pytest.raises(ValueError):
        backend.execute_circuit(circuit, initial_state=np.ones(2))

    initial_circuit = Circuit(1)
    circuit.add(gates.GPI2(0, phi=np.pi / 2))
    backend.execute_circuit(circuit, initial_state=initial_circuit)


@pytest.mark.parametrize(
    "gate,kwargs",
    [
        (gates.I, {}),
        (gates.Z, {}),
        (gates.GPI, {"phi": np.pi / 8}),
        (gates.GPI2, {"phi": np.pi / 8}),
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
    initial_state_circuit = Circuit(3)
    initial_state_circuit.add(gates.GPI(0, phi=np.pi / 2))
    circuit = Circuit(3)
    circuit.add(gates.GPI2(i, phi=np.pi / 2) for i in range(3))
    circuit.add(gates.M(0, 1, 2))

    results = backend.execute_circuits(
        5 * [circuit], initial_states=initial_state_circuit, nshots=100
    )
    assert len(results) == 5
    for result in results:
        assert result.samples().shape == (100, 3)
        assert sum(result.frequencies().values()) == 100


def test_multiple_measurements():
    backend = QibolabBackend("dummy")

    circuit = Circuit(4)
    circuit.add(gates.GPI2(i, phi=np.pi / 2) for i in range(2))
    circuit.add(gates.CZ(1, 2))
    res0 = circuit.add(gates.M(0))
    res1 = circuit.add(gates.M(3))
    res2 = circuit.add(gates.M(1))
    result = backend.execute_circuit(circuit, nshots=50)

    samples = [res.samples()[:, 0] for res in [res0, res1, res2]]
    final_samples = np.array(samples).T
    target_samples = result.samples()
    np.testing.assert_allclose(final_samples, target_samples)


def dummy_string_qubit_names():
    """Create dummy platform with string-named qubits."""
    platform = create_platform("dummy")
    for q, qubit in platform.qubits.items():
        qubit.name = f"A{q}"
    platform.runcard.native_gates.single_qubit = {
        qubit.name: qubit for qubit in platform.qubits.values()
    }
    platform.runcard.native_gates.two_qubit = {
        (f"A{q0}", f"A{q1}"): pair for (q0, q1), pair in platform.pairs.items()
    }
    return platform


def test_execute_circuit_str_qubit_names():
    """Check that platforms with qubits that have non-integer names can execute
    circuits."""
    backend = QibolabBackend(dummy_string_qubit_names())
    circuit = Circuit(3)
    circuit.add(gates.GPI2(i, phi=np.pi / 2) for i in range(2))
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.M(0, 1))
    result = backend.execute_circuit(circuit, nshots=20)
    assert result.samples().shape == (20, 2)


@pytest.mark.qpu
@pytest.mark.xfail(
    raises=AssertionError, reason="Probabilities are not well calibrated"
)
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
@pytest.mark.xfail(
    raises=AssertionError, reason="Probabilities are not well calibrated"
)
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
@pytest.mark.xfail(
    raises=AssertionError, reason="Probabilities are not well calibrated"
)
def test_superposition_for_all_qubits(connected_backend):
    """Applies an H gate to each qubit of the circuit and measures the
    probabilities."""
    nshots = 5000
    nqubits = connected_backend.platform.nqubits
    probs = []
    for q in range(nqubits):
        circuit = Circuit(nqubits)
        circuit.add(gates.GPI2(q=q, phi=np.pi / 2))
        circuit.add(gates.M(q))
        freqs = connected_backend.execute_circuit(circuit, nshots=nshots).frequencies(
            binary=False
        )
        probs.append([freqs[i] / nshots for i in range(2)])
        warnings.warn(
            f"Probabilities after an Hadamard gate applied to qubit {q}: {probs[-1]}"
        )
    probs = np.asarray(probs)
    target_probs = np.repeat(a=0.5, repeats=nqubits)
    np.testing.assert_allclose(probs.T[0], target_probs, atol=0.05)
    np.testing.assert_allclose(probs.T[1], target_probs, atol=0.05)


# TODO: test_circuit_result_tensor
# TODO: test_circuit_result_representation


def test_metabackend_load(platform):
    backend = MetaBackend.load(platform.name)
    assert isinstance(backend, QibolabBackend)
    assert backend.platform.name == platform.name


def test_metabackend_list_available(tmpdir):
    for platform in (
        "valid_platform/platform.py",
        "invalid_platform/invalid_platform.py",
    ):
        path = Path(tmpdir / platform)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    os.environ["QIBOLAB_PLATFORMS"] = str(tmpdir)
    available_platforms = {"valid_platform": True}
    assert MetaBackend().list_available() == available_platforms
