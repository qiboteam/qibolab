"""Unit tests for Qblox result extraction (``results.extract``)."""

import uuid

import numpy as np

from qibolab._core.execution_parameters import AcquisitionType
from qibolab._core.instruments.qblox.results import extract


def _scope_acquired_data(n_samples: int) -> dict:
    """Build an ``AcquiredData`` dict in the nested scope format returned by
    ``qblox_instruments``."""
    acq_name = str(uuid.uuid4())
    return {
        acq_name: {
            "index": 0,
            "acquisition": {
                "scope": {
                    "path0": {
                        "data": list(np.linspace(-1.0, 1.0, n_samples)),
                        "out-of-range": False,
                        "avg_cnt": 1,
                    },
                    "path1": {
                        "data": list(np.linspace(1.0, -1.0, n_samples)),
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


def test_extract_raw_scope_is_numeric():
    """Test that ``extract`` with ``AcquisitionType.RAW`` returns a numeric
    ``(n_samples, 2)`` array from the nested scope data."""
    n_samples = 10
    acquisitions = {"ch0": _scope_acquired_data(n_samples)}
    acq_name = next(iter(acquisitions["ch0"]))

    result = extract(
        acquisitions,
        lengths={uuid.UUID(acq_name): n_samples},
        acquisition=AcquisitionType.RAW,
        shape=(n_samples, 2),
    )

    (value,) = result.values()
    assert value.shape == (n_samples, 2)
    assert value.dtype == np.float64
    # path0 -> I (axis 1 == 0), path1 -> Q (axis 1 == 1)
    np.testing.assert_allclose(value[:, 0], np.linspace(-1.0, 1.0, n_samples))
    np.testing.assert_allclose(value[:, 1], np.linspace(1.0, -1.0, n_samples))
