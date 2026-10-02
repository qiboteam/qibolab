"""Tests for SequencerConfig.build with mixer corrections and channel configurations."""

from qibolab._core.components.channels import AcquisitionChannel, IqChannel
from qibolab._core.components.configs import (
    AcquisitionConfig,
    IqConfig,
    IqMixerConfig,
    OscillatorConfig,
)
from qibolab._core.execution_parameters import AcquisitionType
from qibolab._core.instruments.qblox.config.port import PortAddress
from qibolab._core.instruments.qblox.config.sequencer import SequencerConfig

IF_FREQ = 500e6
GAIN, PHASE = 0.953, 2.75


def _channels_configs():
    """Create a test setup with two IQ channels sharing one RF port and mixer."""
    channels = {
        "0/drive": IqChannel(path="4/o1", lo="lo/0", mixer="mixer/0"),
        "0/drive12": IqChannel(path="4/o1", lo="lo/0", mixer="mixer/0"),
        "0/acquisition": AcquisitionChannel(path="4/i1", probe="0/drive"),
    }
    configs = {
        "0/drive": IqConfig(frequency=4.5e9, scale_q=GAIN, phase_q=PHASE),
        "0/drive12": IqConfig(frequency=4.7e9, scale_q=0.9, phase_q=-4.0),
        "0/acquisition": AcquisitionConfig(delay=0, smearing=0),
        "lo/0": OscillatorConfig(frequency=4.5e9 - IF_FREQ, power=-10),
        "mixer/0": IqMixerConfig(offset_i=0.03, offset_q=-0.05),
    }
    return channels, configs


def _build(channel_id, channels, configs, path):
    """Build a SequencerConfig for a channel at the given port address."""
    return SequencerConfig.build(
        address=PortAddress.from_path(path),
        channel_id=channel_id,
        channels=channels,
        configs=configs,
        acquisition=AcquisitionType.RAW,
        rf=True,
    )


def test_iq_channel_applies_corrections():
    """IQ channels apply their configured mixer corrections and NCO frequency."""
    channels, configs = _channels_configs()
    cfg = _build("0/drive", channels, configs, "4/o1")

    assert cfg.nco_freq == int(IF_FREQ)
    assert cfg.mixer_corr_gain_ratio == GAIN
    assert cfg.mixer_corr_phase_offset_degree == PHASE


def test_multiple_channels_on_shared_port_have_independent_corrections():
    channels, configs = _channels_configs()
    drive = _build("0/drive", channels, configs, "4/o1")
    drive12 = _build("0/drive12", channels, configs, "4/o1")

    assert drive.mixer_corr_gain_ratio == GAIN
    assert drive12.mixer_corr_gain_ratio == 0.9
    assert drive.mixer_corr_phase_offset_degree == PHASE
    assert drive12.mixer_corr_phase_offset_degree == -4.0

    assert drive.nco_freq == int(IF_FREQ)
    # NCO freq is the difference between channel frequency and LO frequency
    drive12_expected_nco = int(
        configs["0/drive12"].frequency - configs["lo/0"].frequency
    )
    assert drive12.nco_freq == drive12_expected_nco


def test_acquisition_channel_inherits_probe_corrections():
    """Acquisition channel applies corrections from its probe channel.

    On QRM-RF, probe and acquisition share a single IO sequencer. The acquisition
    channel inherits the sideband and NCO settings from its probe channel's
    configuration.
    """
    channels, configs = _channels_configs()
    cfg = _build("0/acquisition", channels, configs, "4/i1")

    assert cfg.nco_freq == int(IF_FREQ)
    assert cfg.mixer_corr_gain_ratio == GAIN
    assert cfg.mixer_corr_phase_offset_degree == PHASE
