# -*- coding: utf-8 -*-
import numpy as np
import pytest
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.models import Circuit

from qibolab.transpilers.native import NativeGates


def assert_matrices_allclose(gate):
    backend = NumpyBackend()
    native_gates = NativeGates()
    target_matrix = gate.asmatrix(backend)
    circuit = Circuit(len(gate.qubits))
    circuit.add(native_gates.translate_gate(gate))
    native_matrix = circuit.unitary(backend)
    np.testing.assert_allclose(native_matrix, target_matrix, atol=1e-12)


@pytest.mark.parametrize("gatename", ["H", "X", "Y"])
def test_pauli_to_native(gatename):
    backend = NumpyBackend()
    native_gates = NativeGates()
    gate = getattr(gates, gatename)(0)
    native_gate = native_gates.translate_gate(gate)
    final_matrix = native_gate.asmatrix(backend)
    target_matrix = -1j * gate.asmatrix(backend)
    np.testing.assert_allclose(final_matrix, target_matrix, atol=1e-15)


@pytest.mark.parametrize("gatename", ["RX", "RY", "RZ"])
def test_rotations_to_native(gatename):
    gate = getattr(gates, gatename)(0, theta=0.1)
    assert_matrices_allclose(gate)


def test_u2_to_native():
    gate = gates.U2(0, phi=0.1, lam=0.3)
    assert_matrices_allclose(gate)


def test_u3_to_native():
    gate = gates.U3(0, theta=0.2, phi=0.1, lam=0.3)
    assert_matrices_allclose(gate)


@pytest.mark.parametrize("gatename", ["CNOT", "CZ", "SWAP", "FSWAP"])
def test_two_qubit_to_native(gatename):
    gate = getattr(gates, gatename)(0, 1)
    assert_matrices_allclose(gate)


@pytest.mark.skip
@pytest.mark.parametrize("gatename", ["CRX", "CRY", "CRZ", "CU1"])
def test_controlled_rotations_to_native(gatename):
    gate = getattr(gates, gatename)(0, 1, theta=0.1)
    assert_matrices_allclose(gate)


@pytest.mark.skip
def test_cu2_to_native():
    gate = gates.CU2(0, 1, phi=0.1, lam=0.2)
    assert_matrices_allclose(gate)


@pytest.mark.skip
def test_cu3_to_native():
    gate = gates.CU3(0, 1, theta=0.3, phi=0.1, lam=0.2)
    assert_matrices_allclose(gate)


@pytest.mark.parametrize("gatename", ["RXX", "RYY", "RZZ"])
def test_rnn_to_native(gatename):
    gate = getattr(gates, gatename)(0, 1, theta=0.1)
    assert_matrices_allclose(gate)


def test_unitary_to_native():
    from scipy.linalg import det, expm

    u = np.random.random((2, 2)) + 1j * np.random.random((2, 2))
    # make random matrix unitary
    u = expm(1j * (u + u.T.conj()))
    # transform to SU(2) form
    u = u / np.sqrt(det(u))
    gate = gates.Unitary(u, 0)
    assert_matrices_allclose(gate)
