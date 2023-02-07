"""Testing result.py"""
import numpy as np
import pytest

from qibolab.result import ExecRes, ExecutionResults


def generate_random_result(length=5):
    i = np.random.rand(length)
    q = np.random.rand(length)
    shots = np.random.rand(length)
    return ExecutionResults.from_components(i, q, shots)


def test_standard_constructor():
    """Testing ExecutionResults constructor"""
    test = np.array([(1, 2), (1, 2)], dtype=ExecRes)
    shots = np.random.rand(3)
    ExecutionResults(test, shots)


def test_execution_result_properties():
    """Testing ExecutionResults properties"""
    results = generate_random_result(5)
    np.testing.assert_equal(np.sqrt(results.i**2 + results.q**2), results.msr)
    np.testing.assert_equal(np.angle(results.i + 1.0j * results.q), results.phase)


@pytest.mark.parametrize("state", [0, 1])
def test_to_dict_probability(state):
    """Testing to_dict_probability method"""
    results = generate_random_result(5)
    if state == 0:
        target_dict = {"probability": results.ground_state_probability}
    else:
        target_dict = {"probability": 1 - results.ground_state_probability}

    assert target_dict == results.to_dict_probability(state=state)


@pytest.mark.parametrize("average", [True, False])
def test_to_dict(average):
    """Testing to_dict method"""
    results = generate_random_result(5)
    output = results.to_dict(average=average)
    if not average:
        target_dict = {
            "MSR[V]": results.msr,
            "i[V]": results.i,
            "q[V]": results.q,
            "phase[rad]": results.phase,
        }
        assert output.keys() == target_dict.keys()
        for key in output:
            np.testing.assert_equal(output[key], target_dict[key])
    else:
        avg = results.compute_average()
        target_dict = {
            "MSR[V]": np.sqrt(avg.i**2 + avg.q**2),
            "i[V]": avg.i,
            "q[V]": avg.q,
            "phase[rad]": np.angle(avg.i + 1.0j * avg.q),
        }
        assert results.to_dict(average=average) == target_dict
