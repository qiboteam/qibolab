from types import SimpleNamespace

import pytest

from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.instruments.qblox import cluster


def test_enable_raw_acquisition_scope_averaging(monkeypatch):
    configurations = []
    selected = []

    class ModuleConfig:
        def __init__(self, **kwargs):
            configurations.append(kwargs)

        def update_module(self, module):
            pass

    class ReadoutModule:
        def scope_acq_sequencer_select(self, index):
            selected.append(index)

    monkeypatch.setattr(cluster.config, "ModuleConfig", ModuleConfig)

    cluster.Cluster._enable_raw_acquisition(
        SimpleNamespace(_modules={1: ReadoutModule()}),
        acq_sequencers={1: [0]},
    )

    assert configurations == [
        {
            "ports": {},
            "scope_acq_avg_mode_en_path0": True,
            "scope_acq_avg_mode_en_path1": True,
        }
    ]
    assert selected == [0]


@pytest.mark.parametrize(
    "averaging_mode", [AveragingMode.SINGLESHOT, AveragingMode.SEQUENTIAL]
)
def test_raw_acquisition_rejects_non_cyclic_averaging(averaging_mode):
    options = ExecutionParameters(
        acquisition_type=AcquisitionType.RAW,
        averaging_mode=averaging_mode,
    )

    with pytest.raises(NotImplementedError, match="AveragingMode.CYCLIC"):
        cluster._validate_raw_averaging_mode(options)


def test_raw_acquisition_accepts_cyclic_averaging():
    options = ExecutionParameters(
        acquisition_type=AcquisitionType.RAW,
        averaging_mode=AveragingMode.CYCLIC,
    )

    cluster._validate_raw_averaging_mode(options)


def test_non_raw_acquisition_allows_single_shot_averaging():
    options = ExecutionParameters(
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    cluster._validate_raw_averaging_mode(options)
