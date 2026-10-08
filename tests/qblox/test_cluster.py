from types import SimpleNamespace

import pytest

from qibolab._core.execution_parameters import AveragingMode
from qibolab._core.instruments.qblox import cluster


@pytest.mark.parametrize(
    ("averaging_mode", "expected"),
    [(AveragingMode.SINGLESHOT, False), (AveragingMode.CYCLIC, True)],
)
def test_enable_raw_acquisition_scope_averaging(
    monkeypatch, averaging_mode, expected
):
    configurations = []

    class ModuleConfig:
        def __init__(self, **kwargs):
            configurations.append(kwargs)

        def update_module(self, module):
            pass

    class ReadoutModule:
        def scope_acq_sequencer_select(self, index):
            pass

    monkeypatch.setattr(cluster.config, "ModuleConfig", ModuleConfig)

    cluster.Cluster._enable_raw_acquisition(
        SimpleNamespace(_modules={1: ReadoutModule()}),
        acq_sequencers={1: [0]},
        averaging_mode=averaging_mode,
    )

    assert configurations == [
        {
            "ports": {},
            "scope_acq_avg_mode_en_path0": expected,
            "scope_acq_avg_mode_en_path1": expected,
        }
    ]
