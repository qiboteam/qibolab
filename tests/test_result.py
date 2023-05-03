"""Testing result.py"""
import numpy as np
import pytest

from qibolab.result import AveragedResults, ExecRes, ExecutionResults


def generate_random_result(length=5):
    i = np.random.rand(length)
    q = np.random.rand(length)
    shots = np.random.rand(length)
    return ExecutionResults.from_components(i, q, shots)


def generate_random_avg_result(length=5):
    i = np.random.rand(length)
    q = np.random.rand(length)
    return AveragedResults.from_components(i, q)


def test_standard_constructor():
    """Testing ExecutionResults constructor"""
    test = np.array([(1, 2), (1, 2)], dtype=ExecRes)
    shots = np.random.rand(3)
    ExecutionResults(test, shots)


def test_execution_result_properties():
    """Testing ExecutionResults properties"""
    results = generate_random_result(5)
    np.testing.assert_equal(np.sqrt(results.i**2 + results.q**2), results.measurement)
    np.testing.assert_equal(np.angle(results.i + 1.0j * results.q), results.phase)


@pytest.mark.parametrize("state", [0, 1])
def test_raw_probability(state):
    """Testing raw_probability method"""
    results = generate_random_result(5)
    if state == 0:
        target_dict = {"probability": results.ground_state_probability}
    else:
        target_dict = {"probability": 1 - results.ground_state_probability}

    assert target_dict == results.raw_probability(state=state)


@pytest.mark.parametrize("average", [True, False])
def test_raw(average):
    """Testing to_dict method"""
    results = generate_random_result(5)
    output = results.raw
    if not average:
        target_dict = {
            "MSR[V]": results.measurement,
            "i[V]": results.i,
            "q[V]": results.q,
            "phase[rad]": results.phase,
        }
        assert output.keys() == target_dict.keys()
        for key in output:
            np.testing.assert_equal(output[key], target_dict[key])
    else:
        avg = results.average
        target_dict = {
            "MSR[V]": np.sqrt(avg.i**2 + avg.q**2),
            "i[V]": avg.i,
            "q[V]": avg.q,
            "phase[rad]": np.angle(avg.i + 1.0j * avg.q),
        }
        assert avg.raw == target_dict


def test_results_add():
    """Testing __add__ method of ExecutionResults"""
    res_1 = generate_random_result(10)
    res_2 = generate_random_result(5)

    total = res_1 + res_2
    target_i = np.append(res_1.i, res_2.i)
    target_q = np.append(res_1.q, res_2.q)
    target_shots = np.append(res_1.shots, res_2.shots)

    np.testing.assert_equal(total.i, target_i)
    np.testing.assert_equal(total.q, target_q)
    np.testing.assert_equal(total.shots, target_shots)


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
    output = results.raw

    target_dict = {
        "MSR[V]": np.sqrt(results.i**2 + results.q**2),
        "i[V]": results.i,
        "q[V]": results.q,
        "phase[rad]": np.angle(results.i + 1.0j * results.q),
    }
    assert output.keys() == target_dict.keys()
    for key in output:
        np.testing.assert_equal(output[key], target_dict[key])
