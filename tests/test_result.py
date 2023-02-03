"""Testing result.py"""
import numpy as np
import pytest

from qibolab.result import ExecutionResults


def generate_random_result(length=5):
    i = np.random.rand(length)
    q = np.random.rand(length)
    shots = np.random.rand(length)
    return ExecutionResults.from_components(i, q, shots)


def generate_random_avg_result(length=5):
    i = np.random.rand(length)
    q = np.random.rand(length)
    return AveragedResults(i, q)


def test_standard_constructor():
    """Testing ExecutionResults constructor"""
    i = np.random.rand(4)
    q = np.random.rand(4)
    shots = np.random.rand(4)
    ExecutionResults(i, q, shots)


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
            "MSR[V]": results.msr.ravel(),
            "i[V]": results.i.ravel(),
            "q[V]": results.q.ravel(),
            "phase[rad]": results.phase.ravel(),
        }
        assert output.keys() == target_dict.keys()
        for key in output:
            np.testing.assert_equal(output[key], target_dict[key])
    else:
        target_dict = {
            "MSR[V]": results.msr.mean(),
            "i[V]": results.i.mean(),
            "q[V]": results.q.mean(),
            "phase[rad]": results.phase.mean(),
        }
        assert results.to_dict(average=average) == target_dict


def test_averaged_results_add():
    """Testing __add__ method of AveragedResults"""
    avg_res_1 = generate_random_avg_result(10)
    avg_res_2 = generate_random_avg_result(5)

    total = avg_res_1 + avg_res_2
    target_i = np.append(avg_res_1.i, avg_res_2.i)
    target_q = np.append(avg_res_1.q, avg_res_2.q)

    np.testing.assert_equal(total.i, target_i)
    np.testing.assert_equal(total.q, target_q)


def test_to_dict_averaged_results():
    """Testing to_dict method"""
    results = generate_random_avg_result(5)
    output = results.to_dict()

    target_dict = {
        "MSR[V]": np.sqrt(results.i**2 + results.q**2),
        "i[V]": results.i,
        "q[V]": results.q,
        "phase[rad]": np.angle(results.i + 1.0j * results.q),
    }
    assert output.keys() == target_dict.keys()
    for key in output:
        np.testing.assert_equal(output[key], target_dict[key])
