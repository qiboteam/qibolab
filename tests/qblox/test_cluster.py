from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.instruments.qblox import cluster, validate
from qibolab._core.instruments.qblox.cluster import Cluster
from qibolab._core.instruments.qblox.config import ModuleConfig


@pytest.mark.parametrize("acq_sequencers", [{}, {1: [0]}, {1: [2], 2: [1]}])
def test_enable_raw_acquisition_selects_scope_sequencer(acq_sequencers):
    """Select the scope sequencer of RAW modules, without touching inactive ones."""
    modules = {slot: MagicMock() for slot in (1, 2)}

    Cluster._enable_raw_acquisition(
        SimpleNamespace(_modules=modules), acq_sequencers=acq_sequencers
    )

    for slot, module in modules.items():
        assert module.scope_acq_sequencer_select.call_args_list == [
            call(index) for index in acq_sequencers.get(slot, [])
        ]
        # scope averaging is a static module setting, cf. `ModuleConfig`
        module.parameters.__getitem__.assert_not_called()


def test_enable_raw_acquisition_rejects_multiple_sequencers_per_module():
    """Scope selection is module-wide, so only one sequencer may acquire."""
    modules = {1: MagicMock()}

    with pytest.raises(ValueError, match="only one acquisition sequencer per module"):
        Cluster._enable_raw_acquisition(
            SimpleNamespace(_modules=modules), acq_sequencers={1: [0, 2]}
        )

    modules[1].scope_acq_sequencer_select.assert_not_called()


@pytest.mark.parametrize("qrm", [True, False])
def test_module_config_enables_scope_averaging(qrm):
    """Scope averaging is enabled on both paths, for QRM modules only."""
    module = MagicMock(is_qrm_type=qrm, is_rf_type=True)
    module.parameters = defaultdict(MagicMock)

    ModuleConfig(ports={}).apply(module)

    for name in ("scope_acq_avg_mode_en_path0", "scope_acq_avg_mode_en_path1"):
        if qrm:
            module.parameters[name].set.assert_called_once_with(True)
        else:
            assert name not in module.parameters


def test_raw_acquisition_sequencers():
    """Only sequencers with disabled acquisition demodulation are selected."""
    seqcfgs = {
        1: {
            0: SimpleNamespace(demod_en_acq=None),
            2: SimpleNamespace(demod_en_acq=False),
        },
        2: {0: SimpleNamespace(demod_en_acq=True)},
        3: {},
    }

    assert cluster._raw_acquisition_sequencers(seqcfgs) == {1: [2]}


def test_play_rejects_raw_non_cyclic_before_execution():
    """The averaging mode is validated before touching the instrument."""
    options = ExecutionParameters(
        acquisition_type=AcquisitionType.RAW,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    # not connected: any instrument access would fail with a different error
    with pytest.raises(NotImplementedError, match="AveragingMode.CYCLIC"):
        Cluster(name="cluster", address="0.0.0.0").play({}, [], options, [])


@pytest.mark.parametrize(
    "averaging_mode", [AveragingMode.SINGLESHOT, AveragingMode.SEQUENTIAL]
)
def test_raw_acquisition_rejects_non_cyclic_averaging(averaging_mode):
    """Reject modes that cannot use hardware-averaged scope data."""
    # Scope-memory limits rule out storing separate RAW traces for each shot.
    options = ExecutionParameters(
        acquisition_type=AcquisitionType.RAW,
        averaging_mode=averaging_mode,
    )

    with pytest.raises(NotImplementedError, match="AveragingMode.CYCLIC"):
        validate.validate_raw_averaging_mode(options)


def test_raw_acquisition_accepts_cyclic_averaging():
    """Allow RAW capture when shots are averaged in the scope hardware."""
    options = ExecutionParameters(
        acquisition_type=AcquisitionType.RAW,
        averaging_mode=AveragingMode.CYCLIC,
    )

    validate.validate_raw_averaging_mode(options)


def test_non_raw_acquisition_allows_single_shot_averaging():
    """Keep the scope-memory restriction specific to RAW acquisitions."""
    options = ExecutionParameters(
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    validate.validate_raw_averaging_mode(options)
