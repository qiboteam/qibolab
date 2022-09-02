# -*- coding: utf-8 -*-
import numpy as np
from qibo.config import raise_error

magic_basis = np.array(
    [[1, 0, 0, 1], [-1j, 0, 0, 1j], [0, 1, -1, 0], [0, -1j, -1j, 0]]
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
    if not np.allclose(d, [1, 0]):
        raise_error(
            ValueError,
            f"Unexpected singular values: {d}\n" "Can only decompose product states.",
        )
    return u[:, 0], v[0]


def tensor_product(ua, ub):
    t = np.transpose(np.tensordot(ua, ub, axes=0), [0, 2, 1, 3])
    return np.reshape(t, (4, 4))


def calculate_single_qubit_unitaries(psi):
    """Step (2)."""
    # TODO: Handle the case where psi is not real in the magic basis
    psi_magic = np.dot(magic_basis, psi.T)
    if not np.allclose(psi_magic.imag, np.zeros_like(psi_magic)):
        raise_error(NotImplementedError, "Given state is not real in the magic basis.")
    psi_bar = psi

    # find e and f by inverting (A3), (A4)
    ef = (psi_bar[0] + 1j * psi_bar[1]) / np.sqrt(2)
    e_f_ = (psi_bar[0] - 1j * psi_bar[1]) / np.sqrt(2)
    e, f = schmidt_decompose(ef)
    e_, f_ = schmidt_decompose(e_f_)
    # find exp(1j * delta) using (A5a)
    ef_ = np.tensordot(e, f_, axes=0).ravel()
    phase = 1j * np.sqrt(2) * np.dot(np.conj(ef_), psi_bar[3])

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
    u_dagger = np.conj(tensor_product(ua, ub).T)
    v_dagger = np.conj(tensor_product(va, vb).T)
    return np.dot(np.dot(u_dagger, unitary), v_dagger)
