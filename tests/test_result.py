"""Testing result.py"""
import numpy as np
import pytest

from qibolab.result import (
    AveragedIntegratedResults,
    AveragedRawWaveformResults,
    AveragedResults,
    AveragedSampleResults,
    ExecRes,
    ExecutionResults,
    IntegratedResults,
    RawWaveformResults,
    SampleResults,
)


def generate_random_result(length=5):
    i = np.random.rand(length)
    q = np.random.rand(length)
    shots = np.random.rand(length)
    return ExecutionResults.from_components(i, q, shots)


def generate_random_iq_result(length=5):
    data = np.random.rand(length, length, length)
    return IntegratedResults(data)


def generate_random_raw_result(length=5):
    data = np.random.rand(length, length, length)
    return IntegratedResults(data)


def generate_random_state_result(length=5):
    data = np.random.randint(low=2, size=(length, length, length))
    return SampleResults(data)


def generate_random_avg_result(length=5):
    i = np.random.rand(length)
    q = np.random.rand(length)
    return AveragedResults.from_components(i, q)


def generate_random_avg_iq_result(length=5):
    data = np.random.rand(length, length, length)
    return AveragedIntegratedResults(data)


def generate_random_avg_raw_result(length=5):
    data = np.random.rand(length, length, length)
    return AveragedIntegratedResults(data)


def generate_random_avg_state_result(length=5):
    data = np.random.randint(low=2, size=(length, length, length))
    return AveragedSampleResults(data)


def test_standard_constructor():
    """Testing ExecutionResults constructor"""
    test = np.array([(1, 2), (1, 2)], dtype=ExecRes)
    shots = np.random.rand(3)
    ExecutionResults(test, shots)


def test_iq_constructor():
    """Testing ExecutionResults constructor"""
    test = np.array([(1, 2), (1, 2)], dtype=ExecRes)
    IntegratedResults(test)


def test_raw_constructor():
    """Testing ExecutionResults constructor"""
    test = np.array([(1, 2), (1, 2)], dtype=ExecRes)
    RawWaveformResults(test)


def test_state_constructor():
    """Testing ExecutionResults constructor"""
    test = np.array([1, 1, 0])
    SampleResults(test)


def test_execution_result_properties():
    """Testing ExecutionResults properties"""
    results = generate_random_result(5)
    np.testing.assert_equal(np.sqrt(results.i**2 + results.q**2), results.measurement)
    np.testing.assert_equal(np.angle(results.i + 1.0j * results.q), results.phase)


@pytest.mark.parametrize("result", ["iq", "raw"])
def test_integrated_result_properties(result):
    """Testing IntegratedResults and RawWaveformResults properties"""
    if result == "iq":
        results = generate_random_iq_result(5)
    else:
        results = generate_random_raw_result(5)
    np.testing.assert_equal(np.sqrt(results.voltage_i**2 + results.voltage_q**2), results.magnitude)
    np.testing.assert_equal(np.angle(results.voltage_i + 1.0j * results.voltage_q), results.phase)


@pytest.mark.parametrize("state", [0, 1])
def test_raw_probability(state):
    """Testing raw_probability method"""
    results = generate_random_result(5)
    if state == 0:
        target_dict = {"probability": results.ground_state_probability}
    else:
        target_dict = {"probability": 1 - results.ground_state_probability}

    assert target_dict == results.raw_probability(state=state)


@pytest.mark.parametrize("state", [0, 1])
def test_state_probability(state):
    """Testing raw_probability method"""
    results = generate_random_state_result(5)
    if state == 0:
        target_dict = {"probability": results.probability(0)}
    else:
        target_dict = {"probability": results.probability(1)}

    assert np.allclose(target_dict["probability"], results.probability(state=state), atol=1e-08)


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


@pytest.mark.parametrize("average", [True, False])
@pytest.mark.parametrize("result", ["iq", "raw"])
def test_serialize(average, result):
    """Testing to_dict method"""
    if not average:
        if result == "iq":
            results = generate_random_iq_result(5)
        else:
            results = generate_random_raw_result(5)
        output = results.serialize
        target_dict = {
            "MSR[V]": results.magnitude,
            "i[V]": results.voltage_i,
            "q[V]": results.voltage_q,
            "phase[rad]": results.phase,
        }
        assert output.keys() == target_dict.keys()
        for key in output:
            np.testing.assert_equal(output[key], target_dict[key].flatten())
    else:
        if result == "iq":
            results = generate_random_avg_iq_result(5)
        else:
            results = generate_random_avg_iq_result(5)
        output = results.serialize
        avg = results
        target_dict = {
            "MSR[V]": np.sqrt(avg.voltage_i**2 + avg.voltage_q**2),
            "i[V]": avg.voltage_i,
            "q[V]": avg.voltage_q,
            "phase[rad]": np.angle(avg.voltage_i + 1.0j * avg.voltage_q),
        }
        assert avg.serialize.keys() == target_dict.keys()
        for key in output:
            np.testing.assert_equal(avg.serialize[key], target_dict[key].flatten())


@pytest.mark.parametrize("average", [True, False])
def test_serialize_state(average):
    """Testing to_dict method"""
    if not average:
        results = generate_random_state_result(5)
        output = results.serialize
        target_dict = {
            "state_0": abs(1 - np.mean(results.samples, axis=0)),
        }
        assert output.keys() == target_dict.keys()
        for key in output:
            np.testing.assert_equal(output[key], target_dict[key].flatten())
    else:
        results = generate_random_avg_state_result(5)
        output = results.serialize
        avg = results
        target_dict = {
            "state_0": abs(1 - np.mean(results.samples, axis=0)),
        }
        assert avg.serialize.keys() == target_dict.keys()
        for key in output:
            np.testing.assert_equal(avg.serialize[key], target_dict[key].flatten())


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


@pytest.mark.parametrize("result", ["iq", "raw"])
def test_serialize_averaged_iq_results(result):
    """Testing to_dict method"""
    if result == "iq":
        results = generate_random_avg_iq_result(5)
    else:
        results = generate_random_avg_raw_result(5)
    output = results.serialize
    target_dict = {
        "MSR[V]": np.sqrt(results.voltage_i**2 + results.voltage_q**2),
        "i[V]": results.voltage_i,
        "q[V]": results.voltage_q,
        "phase[rad]": np.angle(results.voltage_i + 1.0j * results.voltage_q),
    }
    assert output.keys() == target_dict.keys()
    for key in output:
        np.testing.assert_equal(output[key], target_dict[key].flatten())


def test_serialize_averaged_state_results():
    """Testing to_dict method"""
    results = generate_random_avg_state_result(5)
    output = results.serialize
    target_dict = {
        "state_0": abs(1 - np.mean(results.samples, axis=0)),
    }
    assert output.keys() == target_dict.keys()
    for key in output:
        np.testing.assert_equal(output[key], target_dict[key].flatten())
