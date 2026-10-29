from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from qibolab._core.components import (
    AcquisitionChannel,
    AcquisitionConfig,
    DcChannel,
    DcConfig,
)
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.instruments.qblox import validate
from qibolab._core.instruments.qblox.cluster import Cluster
from qibolab._core.instruments.qblox.config import ModuleConfig
from qibolab._core.instruments.qblox.sequence import Q1Sequence


@pytest.fixture
def acquisition_cluster():
    instrument = Cluster(
        name="cluster",
        address="0.0.0.0",
        channels={
            "0/drive": DcChannel(path="1/o1"),
            "1/acquisition": AcquisitionChannel(path="1/i1"),
            "2/acquisition": AcquisitionChannel(path="2/i1"),
            "3/acquisition": AcquisitionChannel(path="3/i1"),
        },
    )
    instrument._cluster = MagicMock(
        modules=[
            MagicMock(
                slot_idx=slot,
                is_rf_type=False,
                sequencers=[MagicMock() for _ in range(3)],
            )
            for slot in (1, 2, 3)
        ]
    )
    configs = {
        channel: AcquisitionConfig(delay=0, smearing=0)
        if isinstance(config, AcquisitionChannel)
        else DcConfig(offset=0)
        for channel, config in instrument.channels.items()
    }
    return instrument, configs


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


@pytest.mark.parametrize("exec_mode", [False, True])
def test_configure_rejects_multiple_raw_sequencers_per_module(
    acquisition_cluster, exec_mode
):
    """Scope selection is module-wide, so only one sequencer may acquire."""
    instrument, configs = acquisition_cluster
    instrument.channels["4/acquisition"] = AcquisitionChannel(path="1/i1")
    configs["4/acquisition"] = AcquisitionConfig(delay=0, smearing=0)
    sequences = (
        {channel: Q1Sequence.empty() for channel in instrument.channels}
        if exec_mode
        else None
    )

    with pytest.raises(ValueError, match="only one acquisition sequencer per module"):
        instrument._configure_sequencers(
            configs, acquisition=AcquisitionType.RAW, sequences=sequences
        )


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


@pytest.mark.parametrize("acquisition", list(AcquisitionType))
@pytest.mark.parametrize("exec_mode", [False, True])
def test_raw_acquisition_sequencers(acquisition_cluster, acquisition, exec_mode):
    """Only sequencers with disabled acquisition demodulation are selected."""
    instrument, configs = acquisition_cluster
    sequences = (
        {
            channel: Q1Sequence.empty()
            for channel in instrument.channels
            if channel != "3/acquisition"
        }
        if exec_mode
        else None
    )

    sequencers, seqcfgs, acquisition_sequencers = instrument._configure_sequencers(
        configs, acquisition=acquisition, sequences=sequences
    )

    expected = {1: [1], 2: [0]}
    if not exec_mode:
        expected[3] = [0]
    assert acquisition_sequencers == (
        expected if acquisition is AcquisitionType.RAW else {}
    )
    assert sequencers[1] == {"0/drive": 0, "1/acquisition": 1}
    assert sequencers[2] == {"2/acquisition": 0}
    assert seqcfgs[1][0].demod_en_acq is None
    assert seqcfgs[1][1].demod_en_acq is (acquisition is not AcquisitionType.RAW)
    if exec_mode:
        assert 3 not in sequencers
        assert seqcfgs[3] == {}
        instrument._modules[3].sequencers[0].connect_sequencer.assert_not_called()


def test_play_rejects_raw_single_shot_before_execution():
    """The averaging mode is validated before touching the instrument."""
    options = ExecutionParameters(
        acquisition_type=AcquisitionType.RAW,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    # not connected: any instrument access would fail with a different error
    with pytest.raises(NotImplementedError, match="AveragingMode.SINGLESHOT"):
        Cluster(name="cluster", address="0.0.0.0").play({}, [], options, [])


def test_raw_acquisition_rejects_single_shot_averaging():
    """Reject the only mode that cannot use hardware-averaged scope data."""
    # Scope-memory limits rule out storing separate RAW traces for each shot.
    options = ExecutionParameters(
        acquisition_type=AcquisitionType.RAW,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    with pytest.raises(NotImplementedError, match="AveragingMode.SINGLESHOT"):
        validate.validate_raw_options(options, [])


@pytest.mark.parametrize(
    "averaging_mode", [AveragingMode.CYCLIC, AveragingMode.SEQUENTIAL]
)
def test_raw_acquisition_accepts_averaged_modes(averaging_mode):
    """Allow RAW capture whenever shots are averaged in the scope hardware.

    The Qblox driver executes SEQUENTIAL and CYCLIC identically, so both are
    compatible with a single hardware-averaged trace.
    """
    options = ExecutionParameters(
        acquisition_type=AcquisitionType.RAW,
        averaging_mode=averaging_mode,
    )

    validate.validate_raw_options(options, [])


def test_non_raw_acquisition_allows_single_shot_averaging():
    """Keep the scope-memory restriction specific to RAW acquisitions."""
    options = ExecutionParameters(
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    validate.validate_raw_options(options, [])
