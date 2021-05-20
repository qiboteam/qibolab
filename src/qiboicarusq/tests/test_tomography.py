import pytest
import numpy as np
from qiboicarusq.tomography import Cholesky, Tomography


def test_cholesky_init():
    m = np.random.random((5, 5))
    c = Cholesky.from_matrix(m)
    np.testing.assert_allclose(c.matrix, m)
    v = np.random.random((5,))
    c = Cholesky.from_vector(v)
    np.testing.assert_allclose(c.vector, v)
    with pytest.raises(ValueError):
        c = Cholesky(matrix=m, vector=v)
    with pytest.raises(TypeError):
        c = Cholesky(matrix="test")
    with pytest.raises(TypeError):
        c = Cholesky(vector="test")


def test_cholesky_decompose():
    m = np.array([[1, 2, 3, 4, 5],
                  [2, 3, 4, 5, 6],
                  [3, 4, 5, 6, 7],
                  [4, 5, 6, 7, 8],
                  [5, 6, 7, 8, 9]])
    m = m + m.T
    m = m + 5 * np.eye(5, dtype=m.dtype)
    c = Cholesky.decompose(m)
    target_matrix = np.array([[1, 0, 0, 0, 0],
                              [0, 2, 0, 0, 0],
                              [0, 0, 7, 0, 0],
                              [1, 2, 2, 4, 0],
                              [0, 0, 0, 0, 0]])
    target_vector = np.array([1, 2, 7, 4, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(c.matrix, target_matrix)
    np.testing.assert_allclose(c.vector, target_vector)


def test_cholesky_reconstruct():
    v = np.arange(16)
    c = Cholesky.from_vector(v)
    target_matrix = np.array([
        [0.38709677+0.j, 0.32580645-0.01774194j, 0.21612903-0.02741935j, 0.01693548-0.03145161j],
        [0.32580645+0.01774194j, 0.35564516+0.j, 0.23709677-0.02419355j, 0.01935484-0.03387097j],
        [0.21612903+0.02741935j, 0.23709677+0.02419355j, 0.25+0.j, 0.02177419-0.03629032j],
        [0.01693548+0.03145161j, 0.01935484+0.03387097j, 0.02177419+0.03629032j, 0.00725806+0.j]])
    np.testing.assert_allclose(c.reconstruct(), target_matrix, atol=1e-7)


# TODO: Test tomography class (perhaps using the example?)
