# -*- coding: utf-8 -*-
import numpy as np
from qibo.config import raise_error

magic_basis = np.array(
    [[1, 0, 0, 1], [-1j, 0, 0, 1j], [0, 1, -1, 0], [0, -1j, -1j, 0]]
) / np.sqrt(2)

bell_basis = np.array(
    [[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1, -1, 0]]
) / np.sqrt(2)


def calculate_psi(unitary):
    """Step (1)."""
    # write unitary in magic basis
    u_magic = np.dot(np.dot(magic_basis, unitary), np.conj(magic_basis.T))
    # construct and diagonalize UT_U
    ut_u = np.dot(u_magic.T, u_magic)
    eigvals, psi_magic = np.linalg.eig(ut_u)
    # write psi in computational basis
    psi = np.dot(np.conj(magic_basis.T), psi_magic).T
    return psi, eigvals


def schmidt_decompose(state):
    """Decomposes a two-qubit product state to its single-qubit parts."""
    u, d, v = np.linalg.svd(np.reshape(state, (2, 2)))
    if not np.allclose(d, [1, 0]):  # pragma: no cover
        raise_error(
            ValueError,
            f"Unexpected singular values: {d}\nCan only decompose product states.",
        )
    return u[:, 0], v[0]


def calculate_single_qubit_unitaries(psi):
    """Step (2)."""
    # TODO: Handle the case where psi is not real in the magic basis
    psi_magic = np.dot(magic_basis, psi.T)
    if not np.allclose(psi_magic.imag, np.zeros_like(psi_magic)):  # pragma: no cover
        raise_error(NotImplementedError, "Given state is not real in the magic basis.")
    psi_bar = np.copy(psi)

    # find e and f by inverting (A3), (A4)
    ef = (psi_bar[0] + 1j * psi_bar[1]) / np.sqrt(2)
    e_f_ = (psi_bar[0] - 1j * psi_bar[1]) / np.sqrt(2)
    e, f = schmidt_decompose(ef)
    e_, f_ = schmidt_decompose(e_f_)
    # find exp(1j * delta) using (A5a)
    ef_ = np.kron(e, f_)
    phase = 1j * np.sqrt(2) * np.dot(np.conj(ef_), psi_bar[2])

    # construct unitaries UA, UB using (A6a), (A6b)
    ua = np.tensordot([1, 0], np.conj(e), axes=0) + phase * np.tensordot(
        [0, 1], np.conj(e_), axes=0
    )
    ub = np.tensordot([1, 0], np.conj(f), axes=0) + np.conj(phase) * np.tensordot(
        [0, 1], np.conj(f_), axes=0
    )
    return ua, ub


def calculate_diagonal(unitary, ua, ub, va, vb):
    """Calculates Ud from (A1)."""
    # normalize U_A, U_B, V_A, V_B so that detU_d = 1
    det = np.linalg.det(unitary) ** (1 / 16)
    ua *= det
    ub *= det
    va *= det
    vb *= det
    u_dagger = np.conj(np.kron(ua, ub).T)
    v_dagger = np.conj(np.kron(va, vb).T)
    ud = np.dot(np.dot(u_dagger, unitary), v_dagger)
    return ua, ub, ud, va, vb


def magic_decomposition(unitary):
    psi, eigvals = calculate_psi(unitary)
    psi_tilde = np.conj(np.sqrt(eigvals))[:, np.newaxis] * np.dot(unitary, psi.T).T
    va, vb = calculate_single_qubit_unitaries(psi)
    ua_dagger, ub_dagger = calculate_single_qubit_unitaries(psi_tilde)
    ua, ub = np.conj(ua_dagger.T), np.conj(ub_dagger.T)
    return calculate_diagonal(unitary, ua, ub, va, vb)


def calculate_h_vector(ud):
    ud_bell = np.dot(np.dot(bell_basis, ud), np.conj(bell_basis.T))
    ud_diag = np.diag(ud_bell)
    if not np.allclose(np.diag(ud_diag), ud_bell):  # pragma: no cover
        raise_error(ValueError, "Ud is not diagonal in the Bell basis.")
    uprod = np.prod(ud_diag)
    if not np.allclose(uprod, 1):  # pragma: no cover
        raise_error(ValueError, f"Product of eigenvalues is {uprod} != 1.")

    lambdas = -np.angle(ud_diag)
    hx = (lambdas[0] + lambdas[2]) / 2.0
    hy = (lambdas[1] + lambdas[2]) / 2.0
    hz = (lambdas[0] + lambdas[1]) / 2.0
    return hx, hy, hz


def cnot_decomposition(hx, hy, hz):
    u3 = -1j * (matrices.X + matrices.Z) / np.sqrt(2)
    # use corrected version from PRA paper (not arXiv)
    u2 = -u3 @ expm(-1j * (hx - np.pi / 4) * matrices.X)
    # add an extra exp(-i pi / 4) global phase to get exact match
    v2 = expm(-1j * hz * matrices.Z) * np.exp(-1j * np.pi / 4)
    v3 = expm(1j * hy * matrices.Z)
    w = (matrices.I - 1j * matrices.X) / np.sqrt(2)
    return u2, u3, v2, v3, w
