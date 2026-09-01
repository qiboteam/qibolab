"""Tests for TWPA continuous-waveform (CW) pump support in the Qblox driver."""

from qibolab._core.components import IqChannel, OscillatorConfig
from qibolab._core.instruments.qblox.cluster import Cluster
from qibolab._core.instruments.qblox.config.sequencer import SequencerConfig

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
# `Cluster` instrument attributes
# ---------------------------------------------------------------------------


def test_cluster_twpa_fields_defaults():
    c = Cluster(address="addr", name="my_cluster")
    assert c.twpas == {}
    assert c.turn_off_on_disconnect is True


def test_cluster_twpa_sources_accept_slot_port_paths():
    """Keys of ``twpa_sources`` must be slot-qualified paths (e.g. ``"8/o1"``)."""
    c = Cluster(address="addr", name="my_cluster", twpas={"8/o1": "pump"})
    assert c.twpas == {"8/o1": "pump"}


def test_cluster_turn_off_on_disconnect_flag():
    c = Cluster(address="addr", name="my_cluster", turn_off_on_disconnect=False)
    assert c.turn_off_on_disconnect is False


# ---------------------------------------------------------------------------
# `_configure_twpa` (requires a connected cluster, so here we only validate
# the "no sources" short-circuit and the LO resolution helper)
# ---------------------------------------------------------------------------


def test_configure_twpa_no_sources_is_noop():
    c = Cluster(address="addr", name="my_cluster")
    c._configure_twpa({})  # must not raise


def test_twpa_lo_freq_resolves_from_channel():
    c = Cluster(address="addr", name="my_cluster")
    c.channels = {"8/drive": IqChannel(path="8/o1", lo="lo1")}
    configs = {"lo1": OscillatorConfig(frequency=5.0e9, power=-10.0)}
    from qibolab._core.instruments.qblox.config import PortAddress

    assert c._twpa_lo_freq(8, PortAddress.from_path("8/o1"), configs) == 5_000_000_000


def test_twpa_lo_freq_missing_returns_zero():
    c = Cluster(address="addr", name="my_cluster")
    c.channels = {"8/drive": IqChannel(path="8/o1", lo=None)}
    from qibolab._core.instruments.qblox.config import PortAddress

    assert c._twpa_lo_freq(8, PortAddress.from_path("8/o1"), {}) == 0
