import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from scipy.optimize import minimize
from scipy.linalg import sqrtm
from qibo import gates
from qibo.config import raise_error

class Tomography:
    """
        states (np.ndarray): Array of shape (4**n, n) with the excited state
            population for each pre-rotations; where n is the number of qubits.
        gates (np.ndarray): Optional array of shape (4**n, 2**n, 2**n) with the
            pre-rotations to be used for the linear estimation of the density
            matrix.
    """
    def __init__(self, states, tomo_gates=None):
        self.states = np.multiply.reduce(states, axis=-1)
        # implementation for 4**n rotations
        # we change to just 3 base rotations in the future
        self._gates = tomo_gates
        self._n = int(np.log(len(states)) / np.log(4))
        self._d = np.array(self.gates).shape[-1]

        # results
        self._linear = None
        self._fitres = None
        self._fitrho = None

    @property
    def measurement_operators(self):
        # makes UMU from M
        M = np.zeros(self.gates[0].shape, dtype=complex)
        M[-1, -1] = 1
        return np.array([U.conj().T @ M @ U for U in self.gates])

    @property
    def gates(self):
        if self._gates is None:
            self._gates = self.default_gates()
        return self._gates

    def default_gates(self):
        base_matrices = [g.matrix for g in self._base()]
        return self.tensor_product_combinations(base_matrices, self._n)

    def tensor_product_combinations(self, base, n):
        # n-fold tensor products of the base list
        base = np.array(base)
        index_combinations = self._iter_indices(base, n)
        return [self._kron_all(base[(c,)]) for c in index_combinations]

    @staticmethod
    def gate_sequence(n, base=None):
        if base is None:
            base = np.array(Tomography._base)
        index_combinations = Tomography._iter_indices(base, n)
        g = [base[(c,)] for c in index_combinations]
        for gi in g:
            for i in range(len(gi)):
                gi[i].target_qubits = tuple(i)
        return g

    @staticmethod
    def _base():
        return [gates.I(0), gates.RX(0, np.pi), gates.RX(0, np.pi / 2), gates.RY(0, np.pi / 2)]

    @staticmethod
    def basis_states(n):
        base = [gates.I(0), gates.RX(0)]
        index_combinations = Tomography._iter_indices(base, n)
        g = [base[c] for c in index_combinations]
        for gi in g:
            for i in range(len(gi)):
                gi[i].target_qubits = tuple(i)
        return g

    @staticmethod
    def _iter_indices(iter, n):
        return itertools.product(*[np.arange(len(iter)) for _ in range(n)])

    @staticmethod
    def _kron_all(matrices):
        r = 1
        for m in matrices:
            r = np.kron(r, m)
        return r

    @property
    def linear(self):
        if self._linear is None:
            A = np.tensordot(self.measurement_operators, self.measurement_operators, axes=([1, 2], [2, 1]))
            c = np.linalg.solve(A, self.states)
            self._linear = np.einsum('i,ijk->jk', c, self.measurement_operators, optimize=True)
        return self._linear

    @property
    def fit(self):
        """MLE estimation of the density matrix from given measurements."""
        if self._fitrho is None:
            if self._fitres is None:
                raise_error(ValueError, "Cannot return fitted density matrix "
                                        "before `minimize` is called.")
        return self._fitrho

    @property
    def success(self):
        """Bool that shows if the MLE minimization was successful."""
        if self._fitres is None:
            raise_error(ValueError, "Cannot return minimization success "
                                    "before `minimize` is called.")
        return self._fitres.success

    def minimize(self, tol=1e-5):
        density_matrix = np.copy(self.linear)
        eps = 1e-6  # rounding parameter
        while (np.linalg.eigvalsh(density_matrix) < 0).any():
            density_matrix += eps * np.eye(self._d)
            eps *= 10
        T_initial = np.linalg.cholesky(density_matrix)
        t_real = list(np.real(T_initial[np.tril_indices(self._d, k=0)]))
        t_imag = list(np.imag(T_initial[np.tril_indices(self._d, k=-1)]))
        t_initial = t_real + t_imag

        def _mle(t):
            T = np.zeros((self._d, self._d), dtype=complex)
            T[np.tril_indices(self._d, k=0)] = t[:len(t_real)]
            T[np.tril_indices(self._d, k=-1)] += t[len(t_real):]
            rho = (T.T.conj() @ T) / np.trace(T.T.conj() @ T)
            return np.sum(np.abs((self.states - np.einsum('ijk,kj->i', self.measurement_operators, rho))))

        self._fitres = minimize(_mle, t_initial, tol=tol)

        t = self._fitres.x
        T = np.zeros((self._d, self._d), dtype=complex)
        T[np.tril_indices(self._d, k=0)] = t[:len(t_real)]
        T[np.tril_indices(self._d, k=-1)] += t[len(t_real):]
        self._fitrho = np.matmul(T.T.conj(), T) / np.trace(np.matmul(T.T.conj(), T))

        return self._fitres

    def fidelity(self, theory):
        sqrt_th = sqrtm(theory)
        return abs(np.trace(sqrtm(sqrt_th @ self.fit @ sqrt_th))) * 100
