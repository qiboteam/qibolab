"""Unit tests for the Qblox port- and module-level configuration."""

import pytest

from qibolab._core.components import Channel, Configs, DcChannel, DcConfig
from qibolab._core.identifier import ChannelId
from qibolab._core.instruments.qblox.config.module import ModuleConfig


def test_module_config_dc_offset_end_to_end():
    offset_val = 0.3

    channels: dict[ChannelId, Channel] = {ChannelId("flux"): DcChannel(path="4/o1")}
    configs: Configs = {ChannelId("flux"): DcConfig(offset=offset_val)}

    qcm = ModuleConfig.build(channels, configs, {}, {}, is_qcm_non_rf_type=True)
    assert qcm.ports["out0_offset"] == pytest.approx(offset_val * 2.5)

    with pytest.raises(AssertionError):
        ModuleConfig.build(channels, configs, {}, {}, is_qcm_non_rf_type=False)
