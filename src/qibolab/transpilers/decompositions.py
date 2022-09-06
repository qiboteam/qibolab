# -*- coding: utf-8 -*-
import numpy as np
from qibo.config import raise_error

magic_basis = np.array(
    [[1, 0, 0, 1], [-1j, 0, 0, 1j], [0, 1, -1, 0], [0, -1j, -1j, 0]]
) / np.sqrt(2)

bell_basis = np.array(
    [[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1, -1, 0]]
) / np.sqrt(2)


def u3_decompose(unitary):
    """Decomposes arbitrary one-qubit gates to U3."""
    # https://github.com/Qiskit/qiskit-terra/blob/d2e3340adb79719f9154b665e8f6d8dc26b3e0aa/qiskit/quantum_info/synthesis/one_qubit_decompose.py#L221
    su2 = unitary / np.sqrt(np.linalg.det(unitary))
    theta = 2 * np.arctan2(abs(su2[1, 0]), abs(su2[0, 0]))
    plus = np.angle(su2[1, 1])
    minus = np.angle(su2[1, 0])
    phi = plus + minus
    lam = plus - minus
    return theta, phi, lam


def calculate_psi(unitary):
    """Solves the eigenvalue problem of UT_U.

    See step (1) of Appendix A in arXiv:quant-ph/0011050.

    Args:
        unitary (np.ndarray): Unitary matrix of the gate we are
            decomposing in the computational basis.

    Returns:
        Eigenvectors (in the computational basis) and eigenvalues
        of UT_U.
    """
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
    """Calculates local unitaries that maps a maximally entangled basis to the magic basis.

    See Lemma 1 of Appendix A in arXiv:quant-ph/0011050.

    Args:
        psi (np.ndarray): Maximally entangled two-qubit states that define a basis.

    Returns:
        Local unitaries UA and UB that map the given basis to the magic basis.
    """
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
    """Calculates Ud matrix that can be written as exp(-iH).

    See Eq. (A1) in arXiv:quant-ph/0011050.
    Ud is diagonal in the magic and Bell basis.
    """
    # normalize U_A, U_B, V_A, V_B so that detU_d = 1
    # this is required so that sum(lambdas) = 0
    # and Ud can be written as exp(-iH)
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
    """Decomposes an arbitrary unitary to (A1) from arXiv:quant-ph/0011050."""
    psi, eigvals = calculate_psi(unitary)
    psi_tilde = np.conj(np.sqrt(eigvals))[:, np.newaxis] * np.dot(unitary, psi.T).T
    va, vb = calculate_single_qubit_unitaries(psi)
    ua_dagger, ub_dagger = calculate_single_qubit_unitaries(psi_tilde)
    ua, ub = np.conj(ua_dagger.T), np.conj(ub_dagger.T)
    return calculate_diagonal(unitary, ua, ub, va, vb)


def to_bell_diagonal(ud):
    """Transforms a matrix to the Bell basis and checks if it is diagonal."""
    ud_bell = np.dot(np.dot(bell_basis, ud), np.conj(bell_basis.T))
    ud_diag = np.diag(ud_bell)
    if not np.allclose(np.diag(ud_diag), ud_bell):  # pragma: no cover
        return None
    uprod = np.prod(ud_diag)
    if not np.allclose(uprod, 1):  # pragma: no cover
        return None
    return ud_diag


def calculate_h_vector(ud_diag):
    """Finds h parameters corresponding to exp(-iH).

    See Eq. (4)-(5) in arXiv:quant-ph/0307177.
    """
    lambdas = -np.angle(ud_diag)
    hx = (lambdas[0] + lambdas[2]) / 2.0
    hy = (lambdas[1] + lambdas[2]) / 2.0
    hz = (lambdas[0] + lambdas[1]) / 2.0
    return hx, hy, hz


def cnot_decomposition(hx, hy, hz):
    """Performs decomposition (6) from arXiv:quant-ph/0307177."""
    from qibo import matrices
    from scipy.linalg import expm

    u3 = -1j * (matrices.X + matrices.Z) / np.sqrt(2)
    # use corrected version from PRA paper (not arXiv)
    u2 = -u3 @ expm(-1j * (hx - np.pi / 4) * matrices.X)
    # add an extra exp(-i pi / 4) global phase to get exact match
    v2 = expm(-1j * hz * matrices.Z) * np.exp(-1j * np.pi / 4)
    v3 = expm(1j * hy * matrices.Z)
    w = (matrices.I - 1j * matrices.X) / np.sqrt(2)
    return u2, u3, v2, v3, w


def two_qubit_decomposition(q0, q1, unitary):
    from qibo import gates

    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    gatelist = []
    ud_diag = to_bell_diagonal(unitary)
    if ud_diag is None:
        u4, v4, ud, u1, v1 = magic_decomposition(unitary)
        ud_diag = to_bell_diagonal(ud)
        v1 = H @ v1
        gatelist.extend([gates.Unitary(u1, q0), gates.Unitary(v1, q1)])
    else:
        u4 = np.eye(2, dtype=unitary.dtype)
        v4 = np.eye(2, dtype=unitary.dtype)
        gatelist.append(gates.H(q1))

    hx, hy, hz = calculate_h_vector(ud_diag)
    # TODO: Implement simplified case for hz = 0
    u2, u3, v2, v3, w = cnot_decomposition(hx, hy, hz)
    # change CNOT to CZ using Hadamard gates
    v2 = H @ v2 @ H
    v3 = H @ v3 @ H
    u4 = u4 @ w
    v4 = v4 @ np.conj(w.T) @ H
    gatelist.extend(
        [
            gates.CZ(q0, q1),
            gates.Unitary(u2, q0),
            gates.Unitary(v2, q1),
            gates.CZ(q0, q1),
            gates.Unitary(u3, q0),
            gates.Unitary(v3, q1),
            gates.CZ(q0, q1),
            gates.Unitary(u4, q0),
            gates.Unitary(v4, q1),
        ]
    )
    return gatelist
