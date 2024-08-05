"""Testing result.py."""

import numpy as np
import pytest



@pytest.mark.parametrize("result", ["iq", "raw"])
def test_integrated_result_properties(result):
    """Testing IntegratedResults and RawWaveformResults properties."""
    if result == "iq":
        results = generate_random_iq_result(5)
    else:
        results = generate_random_raw_result(5)
    np.testing.assert_equal(
        np.sqrt(results.voltage_i**2 + results.voltage_q**2), results.magnitude
    )
    np.testing.assert_equal(
        np.unwrap(np.arctan2(results.voltage_i, results.voltage_q)), results.phase
    )


@pytest.mark.parametrize("state", [0, 1])
def test_state_probability(state):
    """Testing raw_probability method."""
    results = generate_random_state_result(5)
    if state == 0:
        target_dict = {"probability": results.probability(0)}
    else:
        target_dict = {"probability": results.probability(1)}

    assert np.allclose(
        target_dict["probability"], results.probability(state=state), atol=1e-08
    )


@pytest.mark.parametrize("average", [True, False])
@pytest.mark.parametrize("result", ["iq", "raw"])
def test_serialize(average, result):
    """Testing to_dict method."""
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
            "phase[rad]": np.unwrap(np.arctan2(avg.voltage_i, avg.voltage_q)),
        }
        assert avg.serialize.keys() == target_dict.keys()
        for key in output:
            np.testing.assert_equal(avg.serialize[key], target_dict[key].flatten())


@pytest.mark.parametrize("average", [True, False])
def test_serialize_state(average):
    """Testing to_dict method."""
    if not average:
        results = generate_random_state_result(5)
        output = results.serialize
        target_dict = {
            "0": abs(1 - np.mean(results.samples, axis=0)),
        }
        assert output.keys() == target_dict.keys()
        for key in output:
            np.testing.assert_equal(output[key], target_dict[key].flatten())
    else:
        results = generate_random_avg_state_result(5)
        assert len(results.serialize["0"]) == 125


@pytest.mark.parametrize("result", ["iq", "raw"])
def test_serialize_averaged_iq_results(result):
    """Testing to_dict method."""
    if result == "iq":
        results = generate_random_avg_iq_result(5)
    else:
        results = generate_random_avg_raw_result(5)
    output = results.serialize
    target_dict = {
        "MSR[V]": np.sqrt(results.voltage_i**2 + results.voltage_q**2),
        "i[V]": results.voltage_i,
        "q[V]": results.voltage_q,
        "phase[rad]": np.unwrap(np.arctan2(results.voltage_i, results.voltage_q)),
    }
    assert output.keys() == target_dict.keys()
    for key in output:
        np.testing.assert_equal(output[key], target_dict[key].flatten())
