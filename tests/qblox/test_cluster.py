from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.instruments.qblox import validate
from qibolab._core.instruments.qblox.config import hardware


@pytest.mark.parametrize("acq_sequencers", [{}, {1: [0]}, {1: [2], 2: [1]}])
def test_enable_raw_acquisition_scope_averaging(acq_sequencers):
    """Enable averaging once per RAW module without touching inactive modules."""
    modules = {
        slot: Mock(
            parameters={
                "scope_acq_avg_mode_en_path0": Mock(),
                "scope_acq_avg_mode_en_path1": Mock(),
            }
        )
        for slot in (1, 2)
    }

    hardware._enable_raw_acquisition(
        SimpleNamespace(_modules=modules),
        acq_sequencers=acq_sequencers,
    )

    for slot, module in modules.items():
        assert module.scope_acq_sequencer_select.call_args_list == [
            call(index) for index in acq_sequencers.get(slot, [])
        ]
        for parameter in module.parameters.values():
            if slot in acq_sequencers:
                parameter.set.assert_called_once_with(True)
            else:
                parameter.set.assert_not_called()


def test_enable_raw_acquisition_rejects_multiple_sequencers_per_module():
    """Scope selection is module-wide, so only one sequencer may acquire."""
    modules = {1: Mock()}

    with pytest.raises(ValueError, match="only one acquisition sequencer per module"):
        hardware._enable_raw_acquisition(
            SimpleNamespace(_modules=modules), acq_sequencers={1: [0, 2]}
        )

    modules[1].scope_acq_sequencer_select.assert_not_called()


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
