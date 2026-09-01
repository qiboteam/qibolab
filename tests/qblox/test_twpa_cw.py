"""Tests for TWPA continuous-waveform (CW) pump support in the Qblox driver."""

import pytest

from qibolab._core.components.configs import OscillatorConfig
from qibolab._core.instruments.qblox.components import QbloxClusterConfig
from qibolab._core.instruments.qblox.config.sequencer import SequencerConfig
from qibolab._core.parameters import ConfigKinds, Parameters

# ---------------------------------------------------------------------------
# `SequencerConfig.build_cw`
# ---------------------------------------------------------------------------


def test_build_cw_basic():
    cfg = SequencerConfig.build_cw(address="out1", nco_freq=1_360_000_000)

    assert cfg.address == "out1"
    assert cfg.cont_mode_en_awg_path0 is True
    assert cfg.cont_mode_en_awg_path1 is True
    assert cfg.cont_mode_waveform_idx_awg_path0 == 0
    assert cfg.cont_mode_waveform_idx_awg_path1 == 0
    assert cfg.nco_freq == 1_360_000_000
    assert cfg.nco_phase_offs == 0.0
    assert cfg.mod_en_awg is True
    assert cfg.sync_en is False
    assert cfg.offset_awg_path0 == 0.5
    assert cfg.offset_awg_path1 == 0.0
    assert cfg.gain_awg_path0 == 1.0
    assert cfg.gain_awg_path1 == 1.0
    assert cfg.marker_ovr_en is True
    assert cfg.marker_ovr_value == 15

    seq = cfg.sequence
    assert seq is not None
    assert seq["program"] == "stop"
    assert "cw" in seq["waveforms"]
    assert len(seq["waveforms"]["cw"]["data"]) % 4 == 0


def test_build_cw_amplitude_override():
    cfg = SequencerConfig.build_cw(address="out0", nco_freq=500_000_000, amplitude=0.25)
    assert cfg.offset_awg_path0 == 0.25
    assert cfg.offset_awg_path1 == 0.0


def test_build_cw_serialization():
    import json

    cfg = SequencerConfig.build_cw(address="out0", nco_freq=1_000_000_000)
    loaded = json.loads(cfg.model_dump_json())
    assert loaded["cont_mode_en_awg_path0"] is True
    assert loaded["nco_freq"] == 1_000_000_000


# ---------------------------------------------------------------------------
# `QbloxClusterConfig`
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _register_qblox_cluster_config():
    """Ensure `QbloxClusterConfig` is registered in `ConfigKinds`."""
    ConfigKinds.reset()
    ConfigKinds.extend([QbloxClusterConfig])
    yield
    ConfigKinds.reset()


def test_qblox_cluster_config_deserialization():
    raw = {
        "kind": "qblox-cluster",
        "twpa_sources": {"o1": "twpa_pump"},
        "turn_off_on_disconnect": False,
    }
    cfg = ConfigKinds.adapted().validate_python(raw)
    assert isinstance(cfg, QbloxClusterConfig)
    assert cfg.kind == "qblox-cluster"
    assert cfg.twpa_sources == {"o1": "twpa_pump"}
    assert cfg.turn_off_on_disconnect is False


def test_qblox_cluster_config_defaults():
    cfg = QbloxClusterConfig()
    assert cfg.twpa_sources == {}
    assert cfg.turn_off_on_disconnect is True


def test_qblox_cluster_config_in_parameters():
    """`QbloxClusterConfig` should be resolvable when mixed with built-in configs."""
    p = Parameters(
        configs={
            "my_cluster": {
                "kind": "qblox-cluster",
                "twpa_sources": {"o1": "twpa"},
            },
            "twpa": {"kind": "oscillator", "frequency": 6.36e9, "power": -10.0},
        }
    )
    assert isinstance(p.configs["my_cluster"], QbloxClusterConfig)
    assert isinstance(p.configs["twpa"], OscillatorConfig)
    assert p.configs["my_cluster"].twpa_sources == {"o1": "twpa"}
