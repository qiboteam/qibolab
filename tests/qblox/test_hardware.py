"""Tests for the Qblox cluster hardware configuration functions."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from qibolab._core.components.channels import AcquisitionChannel, IqChannel
from qibolab._core.components.configs import (
    AcquisitionConfig,
    IqConfig,
    OscillatorConfig,
)
from qibolab._core.execution_parameters import AcquisitionType
from qibolab._core.instruments.qblox.cluster import Cluster
from qibolab._core.instruments.qblox.config import hardware


def _module(slot: int, qrm: bool) -> MagicMock:
    return MagicMock(
        slot_idx=slot,
        is_qrm_type=qrm,
        is_qcm_type=not qrm,
        is_rf_type=True,
        sequencers=[MagicMock() for _ in range(6)],
    )


@pytest.fixture
def cluster() -> Cluster:
    """Build a cluster with a QCM-RF and a QRM-RF module, backed by mocks."""
    cluster = Cluster(
        name="cluster",
        address="0.0.0.0",
        channels={
            "0/drive": IqChannel(path="2/o1", lo="lo/0"),
            "1/drive": IqChannel(path="2/o2", lo="lo/1"),
            "0/probe": IqChannel(path="4/o1", lo="lo/probe"),
            "0/acquisition": AcquisitionChannel(path="4/i1", probe="0/probe"),
        },
    )
    cluster._cluster = SimpleNamespace(modules=[_module(2, False), _module(4, True)])
    return cluster


@pytest.fixture
def configs() -> dict:
    return {
        "0/drive": IqConfig(frequency=4.1e9),
        "1/drive": IqConfig(frequency=4.2e9),
        "lo/0": OscillatorConfig(frequency=4.0e9, power=0),
        "lo/1": OscillatorConfig(frequency=4.0e9, power=0),
        "0/probe": IqConfig(frequency=7.2e9),
        "0/acquisition": AcquisitionConfig(delay=0, smearing=0),
        "lo/probe": OscillatorConfig(frequency=7.0e9, power=0),
    }


def test_configure_modules_static(cluster, configs):
    """Each module is configured and its configuration returned by slot."""
    modcfgs = hardware.configure_modules_static(cluster, configs)

    assert set(modcfgs) == {2, 4}
    for slot, module in cluster._modules.items():
        module.disconnect_outputs.assert_called_once()
        assert len(modcfgs[slot].ports) > 0
    cluster._modules[4].disconnect_inputs.assert_called_once()
    cluster._modules[2].disconnect_inputs.assert_not_called()


def test_configure_hardware_all_channels(cluster, configs):
    """Without sequences, all non-probe channels are assigned to sequencers."""
    sequencers = hardware.configure_hardware_per_sequence(
        cluster, configs, AcquisitionType.INTEGRATION
    )

    assert sequencers == {2: {"0/drive": 0, "1/drive": 1}, 4: {"0/acquisition": 0}}


def test_configure_hardware_active_channels(cluster, configs):
    """With sequences, only the involved channels are configured."""
    sequencers = hardware.configure_hardware_per_sequence(
        cluster,
        configs,
        AcquisitionType.INTEGRATION,
        sequences={"1/drive": None},
    )

    assert sequencers == {2: {"1/drive": 1}}
    cluster._modules[2].sequencers[0].connect_sequencer.assert_not_called()
    cluster._modules[2].sequencers[1].connect_sequencer.assert_called_once()


def test_deprecated_configure(cluster, configs):
    """The deprecated public method keeps its signature and return value."""
    sequencers, cfgs = cluster.configure(configs, AcquisitionType.RAW)

    assert sequencers == {2: {"0/drive": 0, "1/drive": 1}, 4: {"0/acquisition": 0}}
    assert set(cfgs.modules) == {2, 4}
    assert {slot: set(seqs) for slot, seqs in cfgs.sequencers.items()} == {
        2: {0, 1},
        4: {0},
    }
    assert cfgs.sequencers[4][0].demod_en_acq is False


def test_configure_hardware_raw_selects_scope_sequencer(cluster, configs):
    """RAW acquisitions select the acquisition sequencer for scope capture."""
    hardware.configure_hardware_per_sequence(cluster, configs, AcquisitionType.RAW)

    cluster._modules[4].scope_acq_sequencer_select.assert_called_once_with(0)
    cluster._modules[2].scope_acq_sequencer_select.assert_not_called()


def test_configure_hardware_integration_skips_scope(cluster, configs):
    """Non-RAW acquisitions leave the scope sequencer selection untouched."""
    hardware.configure_hardware_per_sequence(
        cluster, configs, AcquisitionType.INTEGRATION
    )

    for module in cluster._modules.values():
        module.scope_acq_sequencer_select.assert_not_called()
