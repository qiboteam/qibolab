"""Unit tests for Qblox result extraction (``results.extract``)."""

import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from qibolab._core.components.channels import AcquisitionChannel, IqChannel
from qibolab._core.components.configs import IqConfig, OscillatorConfig
from qibolab._core.execution_parameters import AcquisitionType
from qibolab._core.instruments.qblox.cluster import Cluster
from qibolab._core.instruments.qblox.results import extract
from qibolab._core.pulses.modulation import modulate


def _scope_acquired_data(path0: list[float], path1: list[float]) -> dict:
    """Build an ``AcquiredData`` dict in the nested scope format returned by
    ``qblox_instruments``."""
    acq_name = str(uuid.uuid4())
    return {
        acq_name: {
            "index": 0,
            "acquisition": {
                "scope": {
                    "path0": {
                        "data": path0,
                        "out-of-range": False,
                        "avg_cnt": 1,
                    },
                    "path1": {
                        "data": path1,
                        "out-of-range": False,
                        "avg_cnt": 1,
                    },
                },
                "bins": {
                    "integration": {"path0": [], "path1": []},
                    "threshold": [],
                    "valid": [],
                    "avg_cnt": [],
                },
            },
        }
    }


def _bins_acquired_data(path0: list, path1: list, threshold: list) -> dict:
    """Build an ``AcquiredData`` dict with populated ``bins`` (integration /
    threshold) data."""
    acq_name = str(uuid.uuid4())
    return {
        acq_name: {
            "index": 0,
            "acquisition": {
                "scope": {
                    "path0": {"data": [], "out-of-range": False, "avg_cnt": 0},
                    "path1": {"data": [], "out-of-range": False, "avg_cnt": 0},
                },
                "bins": {
                    "integration": {"path0": path0, "path1": path1},
                    "threshold": threshold,
                    "valid": [],
                    "avg_cnt": [],
                },
            },
        }
    }


# An explicit zero intermediate frequency must still trigger the sqrt(2) scaling of the demodulation, unlike a missing one.
@pytest.mark.parametrize("frequency", [0.0, 0.1])
def test_extract_raw_scope_is_numeric(frequency):
    """Test that ``extract`` with ``AcquisitionType.RAW`` returns a numeric
    ``(n_samples, 2)`` array from the nested scope data."""
    n_samples = 10
    sampling_rate = 1.0
    # Opposite I/Q values keep the global mean zero, including at zero frequency,
    # so demodulation's DC removal does not change the expected envelope.
    envelope = np.array([np.ones(n_samples), -np.ones(n_samples)])
    modulated = modulate(envelope, frequency, sampling_rate)
    acquisitions = {
        "ch0": _scope_acquired_data(modulated[0].tolist(), modulated[1].tolist())
    }
    acq_name = next(iter(acquisitions["ch0"]))

    result = extract(
        acquired_data=acquisitions,
        lengths={uuid.UUID(acq_name): n_samples},
        acquisition=AcquisitionType.RAW,
        shape=(n_samples, 2),
        sampling_rate=sampling_rate,
        frequencies={"ch0": frequency},
    )

    (value,) = result.values()
    assert value.shape == (n_samples, 2)
    assert value.dtype == np.float64
    np.testing.assert_allclose(
        value.T,
        envelope,
        atol=1e-8,
    )


# Frequencies belonging to other channels must not affect this channel's data.
@pytest.mark.parametrize("frequencies", [{}, {"other_channel": 0.1}])
def test_extract_raw_scope_without_frequency(frequencies):
    """Preserve both scope paths unchanged when the channel has no frequency."""
    # A nonzero mean makes unintended demodulation and DC removal detectable.
    path0 = [1.0, 2.0, 3.0]
    path1 = [-4.0, 5.0, 6.0]
    acquisitions = {"ch0": _scope_acquired_data(path0, path1)}
    acq_id = uuid.UUID(next(iter(acquisitions["ch0"])))

    result = extract(
        acquired_data=acquisitions,
        lengths={acq_id: len(path0)},
        acquisition=AcquisitionType.RAW,
        shape=(len(path0), 2),
        sampling_rate=1.0,
        frequencies=frequencies,
    )

    assert set(result) == {acq_id}
    assert result[acq_id].shape == (len(path0), 2)
    assert result[acq_id].dtype == np.float64
    np.testing.assert_array_equal(result[acq_id], np.column_stack((path0, path1)))


def test_extract_integration():
    """Test that ``extract`` with ``AcquisitionType.INTEGRATION`` returns the
    integrated values divided by the integration length, shaped ``(2,)``."""
    path0, path1 = [2.0], [1.0]
    length = 2
    acquisitions = {"ch0": _bins_acquired_data(path0, path1, threshold=[])}
    acq_name = next(iter(acquisitions["ch0"]))

    result = extract(
        acquired_data=acquisitions,
        lengths={uuid.UUID(acq_name): length},
        acquisition=AcquisitionType.INTEGRATION,
        shape=(2,),
        sampling_rate=1.0,
        frequencies={},
    )

    (value,) = result.values()
    assert value.shape == (2,)
    assert value.dtype == np.float64
    np.testing.assert_allclose(value, [1.0, 0.5])


def test_extract_discrimination():
    """Test that ``extract`` with ``AcquisitionType.DISCRIMINATION`` returns the
    thresholded value, shaped ``()``."""
    threshold = [0]
    acquisitions = {"ch0": _bins_acquired_data(path0=[], path1=[], threshold=threshold)}
    acq_name = next(iter(acquisitions["ch0"]))

    result = extract(
        acquired_data=acquisitions,
        lengths={uuid.UUID(acq_name): 1},
        acquisition=AcquisitionType.DISCRIMINATION,
        shape=(),
        sampling_rate=1.0,
        frequencies={},
    )

    (value,) = result.values()
    assert value.shape == ()
    np.testing.assert_array_equal(value, np.array(0))


def test_cluster_intermediate_frequencies():
    """RAW demodulation uses the probe frequency, relative to the LO when present."""
    channels = {
        "0/probe": IqChannel(path="4/o1", lo="lo/0"),
        "0/acquisition": AcquisitionChannel(path="4/i1", probe="0/probe"),
        "1/probe": IqChannel(path="5/o1"),
        "1/acquisition": AcquisitionChannel(path="5/i1", probe="1/probe"),
        "2/acquisition": AcquisitionChannel(path="6/i1"),
    }
    configs = {
        "0/probe": IqConfig(frequency=7.2e9),
        "lo/0": OscillatorConfig(frequency=7.0e9, power=0),
        "1/probe": IqConfig(frequency=150e6),
    }
    cluster = SimpleNamespace(channels=channels)
    cluster._los = Cluster._los.func(cluster)

    frequencies = Cluster._ifs(cluster, configs)

    assert frequencies == pytest.approx(
        {"0/acquisition": 200e6, "1/acquisition": 150e6}
    )
