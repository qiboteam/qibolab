import os
import json
import pathlib
import pytest
import numpy as np
from qibolab.tomography import Tomography

REGRESSION_FOLDER = pathlib.Path(__file__).with_name("regressions")


def assert_regression_fixture(array, filename):
    """Check array matches data inside filename.

    Args:
        array: numpy array
        filename: fixture filename

    If filename does not exists, this function
    creates the missing file otherwise it loads
    from file and compare.
    """
    filename = REGRESSION_FOLDER/filename
    try:
        target = np.load(filename)
        np.testing.assert_allclose(array, target)
    except: # pragma: no cover
        # case not tested in GitHub workflows because files exist
        np.save(filename, array)


# def test_cholesky_init():
#     m = np.random.random((5, 5))
#     c = Cholesky.from_matrix(m)
#     np.testing.assert_allclose(c.matrix, m)
#     v = np.random.random((5,))
#     c = Cholesky.from_vector(v)
#     np.testing.assert_allclose(c.vector, v)
#     with pytest.raises(ValueError):
#         c = Cholesky(matrix=m, vector=v)
#     with pytest.raises(TypeError):
#         c = Cholesky(matrix="test")
#     with pytest.raises(TypeError):
#         c = Cholesky(vector="test")


# def test_cholesky_decompose():
#     m = np.array([[1, 2, 3, 4, 5],
#                   [2, 3, 4, 5, 6],
#                   [3, 4, 5, 6, 7],
#                   [4, 5, 6, 7, 8],
#                   [5, 6, 7, 8, 9]])
#     m = m + m.T
#     m = m + 5 * np.eye(5, dtype=m.dtype)
#     c = Cholesky.decompose(m)
#     target_matrix = np.array([[1, 0, 0, 0, 0],
#                               [0, 2, 0, 0, 0],
#                               [0, 0, 7, 0, 0],
#                               [1, 2, 2, 4, 0],
#                               [0, 0, 0, 0, 0]])
#     target_vector = np.array([1, 2, 7, 4, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0,
#                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#     np.testing.assert_allclose(c.matrix, target_matrix)
#     np.testing.assert_allclose(c.vector, target_vector)


# def test_cholesky_reconstruct():
#     v = np.arange(16)
#     c = Cholesky.from_vector(v)
#     target_matrix = np.array([
#         [0.38709677+0.j, 0.32580645-0.01774194j, 0.21612903-0.02741935j, 0.01693548-0.03145161j],
#         [0.32580645+0.01774194j, 0.35564516+0.j, 0.23709677-0.02419355j, 0.01935484-0.03387097j],
#         [0.21612903+0.02741935j, 0.23709677+0.02419355j, 0.25+0.j, 0.02177419-0.03629032j],
#         [0.01693548+0.03145161j, 0.01935484+0.03387097j, 0.02177419+0.03629032j, 0.00725806+0.j]])
#     np.testing.assert_allclose(c.reconstruct(), target_matrix, atol=1e-7)


# def test_tomography_find_beta():
#     amplitudes = np.random.random(16)
#     state = np.array([1, 2, 3, 4])
#     tom = Tomography(amplitudes, state)
#     target_beta = [2.5, -1, -0.5, 0]
#     np.testing.assert_allclose(tom.find_beta(state), target_beta)


@pytest.mark.skip
def test_tomography_init():
    n = 3
    states = np.random.random((4**n, n))
    gates = np.random.random((4**n, 2**n, 2**n))
    tom = Tomography(states, gates)
    np.testing.assert_allclose(tom.states, states)
    np.testing.assert_allclose(tom.gates, gates)


def test_tomography_default_gates():
    n = 3
    states = np.random.random((4**n, n))
    tom = Tomography(states)
    assert_regression_fixture(tom.gates, "default_gates.npy")


def test_tomography_linear():
    n = 3
    states = np.random.random((4**n, n))
    tom = Tomography(states)
    assert_regression_fixture(tom.linear, "linear_estimation.npy")


@pytest.mark.skip
def test_tomography_fit():
    n = 3
    states = np.random.random((4**n, n))
    tom = Tomography(states)
    with pytest.raises(ValueError):
        tom.fit

    tom.minimize()
    assert tom.success
    assert_regression_fixture(tom.fit, "mlefit_estimation.npy")


def extract_json(filepath):
    with open(filepath, "r") as file:
        raw = json.loads(file.read())
    data = np.stack(list(raw.values()))
    return np.sqrt((data ** 2).sum(axis=1))


@pytest.mark.skip
@pytest.mark.parametrize("state_value,target_fidelity",
                         [(0, 93.01278047175582), (1, 82.30795926024483),
                          (2, 65.06114271984393), (3, 22.230579223385284)])
def test_tomography_example(state_value, target_fidelity):
    state_path = REGRESSION_FOLDER / "states_181120.json"
    amplitude_path = "tomo_181120-{0:02b}.json".format(state_value)
    amplitude_path = REGRESSION_FOLDER / amplitude_path
    state = extract_json(state_path)
    amp = extract_json(amplitude_path)
    tom = Tomography(amp, state)
    tom.minimize()
    assert tom.success
    rho_theory = np.zeros((4, 4), dtype=complex)
    rho_theory[state_value, state_value] = 1
    fidelity = tom.fidelity(rho_theory)
    np.testing.assert_allclose(fidelity, target_fidelity)
