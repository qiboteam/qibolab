"""Tests for TWPA continuous-waveform (CW) sequencer configuration."""

from qibolab._core.components.configs import OscillatorConfig
from qibolab._core.instruments.qblox.config.port import PortAddress
from qibolab._core.instruments.qblox.config.sequencer import SequencerConfig


def test_build_cw_basic():
    """Verify CW sequencer config has expected fields."""
    address = PortAddress.from_path("0/o1")
    osc = OscillatorConfig(frequency=6.36e9, power=-10.0)
    lo_freq = 5.0e9

    cfg = SequencerConfig.build_cw(address=address, osc_config=osc, lo_freq=lo_freq)

    assert cfg.address == "out0"
    assert cfg.cont_mode_en_awg_path0 is True
    assert cfg.cont_mode_en_awg_path1 is True
    assert cfg.cont_mode_waveform_idx_awg_path0 == 0
    assert cfg.cont_mode_waveform_idx_awg_path1 == 0
    assert cfg.nco_freq == int(6.36e9 - 5.0e9)
    assert cfg.nco_phase_offs == 0.0
    assert cfg.mod_en_awg is True
    assert cfg.sync_en is False
    assert cfg.offset_awg_path0 == 0.5
    assert cfg.offset_awg_path1 == 0.0
    assert cfg.gain_awg_path0 == 1.0
    assert cfg.gain_awg_path1 == 1.0
    assert cfg.marker_ovr_en is True
    assert cfg.marker_ovr_value == 15

    # sequence should be a valid CW sequence
    seq = cfg.sequence
    assert seq is not None
    assert seq["program"] == "stop"
    assert "cw" in seq["waveforms"]
    assert len(seq["waveforms"]["cw"]["data"]) % 4 == 0


def test_build_cw_nco_frequency():
    """Verify NCO frequency is computed as absolute - LO."""
    address = PortAddress.from_path("5/o2")
    osc = OscillatorConfig(frequency=7.5e9, power=-5.0)
    lo_freq = 7.0e9

    cfg = SequencerConfig.build_cw(address=address, osc_config=osc, lo_freq=lo_freq)
    assert cfg.nco_freq == 500_000_000  # 0.5 GHz IF


def test_build_cw_port_address():
    """Verify port address mapping."""
    addr = PortAddress.from_path("3/o4")
    osc = OscillatorConfig(frequency=1.0e9, power=0.0)
    cfg = SequencerConfig.build_cw(address=addr, osc_config=osc, lo_freq=500_000_000)
    assert cfg.address == "out3"


def test_build_cw_serialization():
    """Verify CW config can be serialized to JSON."""
    import json

    address = PortAddress.from_path("0/o1")
    osc = OscillatorConfig(frequency=6.0e9, power=-10.0)
    cfg = SequencerConfig.build_cw(address=address, osc_config=osc, lo_freq=5.0e9)

    dumped = cfg.model_dump_json()
    loaded = json.loads(dumped)
    assert loaded["cont_mode_en_awg_path0"] is True
    assert loaded["nco_freq"] == 1_000_000_000
