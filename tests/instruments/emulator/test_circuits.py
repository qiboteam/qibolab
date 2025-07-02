"""Testing basic circuits with emulator platforms."""

import numpy as np
import pytest
from qibo import Circuit, construct_backend, gates


def test_measurement(platform):
    backend = construct_backend(backend="qibolab", platform=platform)
    circuit = Circuit(1)
    circuit.add(gates.M(0))
    result = backend.execute_circuit(circuit, nshots=1000)
    assert pytest.approx(result.samples().mean(), abs=5e-2) == 0


def test_hadamard(platform):
    backend = construct_backend(backend="qibolab", platform=platform)
    circuit = Circuit(1)
    circuit.add(gates.GPI2(0, 0))
    circuit.add(gates.M(0))
    result = backend.execute_circuit(circuit, nshots=1000)
    assert pytest.approx(result.samples().mean(), abs=1e-1) == 0.5


def test_rz(platform):
    backend = construct_backend(backend="qibolab", platform=platform)
    circuit = Circuit(1)
    circuit.add(gates.GPI2(0, 0))
    circuit.add(gates.RZ(0, np.pi / 2))
    circuit.add(gates.GPI2(0, np.pi / 2))
    circuit.add(gates.M(0))
    result = backend.execute_circuit(circuit, nshots=1000)
    assert pytest.approx(result.samples().mean(), abs=1e-1) == 1


@pytest.mark.parametrize("setup", ["Id", "X"])
def test_cnot(platform, setup):
    backend = construct_backend(backend="qibolab", platform=platform)
    if backend.platform.nqubits < 2:
        pytest.skip("CNOT requires at least two qubits.")
    if len(backend.platform.couplers) > 0:
        pytest.skip(f"Plaform {platform} with couplers unsupported.")
    if backend.platform.natives.two_qubit[0, 1].CNOT is None:
        pytest.skip(f"Platform {platform} doesn't support CNOT.")
    circuit = Circuit(2)
    if setup == "X":
        circuit.add(gates.GPI(0, 0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.M(0, 1))
    result = backend.execute_circuit(circuit, nshots=1000)
    assert (
        pytest.approx(
            result.frequencies()["00" if setup == "Id" else "11"] / 1000, abs=3e-1
        )
        == 1
    )


def test_iswap(platform):
    backend = construct_backend(backend="qibolab", platform=platform)
    if backend.platform.nqubits < 2:
        pytest.skip("iSWAP requires at least two qubits.")
    if backend.platform.natives.two_qubit[0, 1].iSWAP is None:
        pytest.skip(f"Platform {platform} doesn't support iSWAP.")
    circuit = Circuit(2)
    circuit.add(gates.GPI(0, 0))
    circuit.add(gates.iSWAP(0, 1))
    circuit.add(gates.M(0, 1))
    result = backend.execute_circuit(circuit, nshots=1000)
    assert pytest.approx(result.frequencies()["01"] / 1000, abs=1e-1) == 1
