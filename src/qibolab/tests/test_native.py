# -*- coding: utf-8 -*-
import numpy as np
import pytest
from qibo import gates

from qibolab.backends import QibolabBackend


def assert_matrices_allclose(backend, native_gate, gate):
    native_matrix = native_gate.asmatrix(backend)
    target_matrix = gate.asmatrix(backend)
    np.testing.assert_allclose(native_matrix, target_matrix)


def test_u3_decomposition():
    backend = QibolabBackend(platform="tii1q")
    theta, phi, lam = 0.1, 0.2, 0.3
    u3_matrix = gates.U3(0, theta, phi, lam).asmatrix(backend)
    rz1 = gates.RZ(0, phi).asmatrix(backend)
    rz2 = gates.RZ(0, theta).asmatrix(backend)
    rz3 = gates.RZ(0, lam).asmatrix(backend)
    rx1 = gates.RX(0, -np.pi / 2).asmatrix(backend)
    rx2 = gates.RX(0, np.pi / 2).asmatrix(backend)
    target_matrix = rz1 @ rx1 @ rz2 @ rx2 @ rz3
    np.testing.assert_allclose(u3_matrix, target_matrix)


@pytest.mark.parametrize("gatename", ["H", "X", "Y"])
def test_pauli_to_native(gatename):
    backend = QibolabBackend(platform="tii1q")
    gate = getattr(gates, gatename)(0)
    native_gate = backend.asnative(gate)
    final_matrix = native_gate.asmatrix(backend)
    target_matrix = -1j * gate.asmatrix(backend)
    np.testing.assert_allclose(final_matrix, target_matrix, atol=1e-15)


@pytest.mark.parametrize("gatename", ["RX", "RY", "RZ"])
def test_rotations_to_native(gatename):
    backend = QibolabBackend(platform="tii1q")
    gate = getattr(gates, gatename)(0, theta=0.1)
    native_gate = backend.asnative(gate)
    assert_matrices_allclose(backend, native_gate, gate)


def test_u2_to_native():
    backend = QibolabBackend(platform="tii1q")
    gate = gates.U2(0, phi=0.1, lam=0.3)
    native_gate = backend.asnative(gate)
    assert_matrices_allclose(backend, native_gate, gate)


def test_unitary_to_native():
    from scipy.linalg import det, expm

    backend = QibolabBackend(platform="tii1q")
    u = np.random.random((2, 2)) + 1j * np.random.random((2, 2))
    # make random matrix unitary
    u = expm(1j * (u + u.T.conj()))
    # transform to SU(2) form
    u = u / np.sqrt(det(u))
    gate = gates.Unitary(u, 0)
    native_gate = backend.asnative(gate)
    assert_matrices_allclose(backend, native_gate, gate)
