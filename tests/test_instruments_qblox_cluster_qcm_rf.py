import numpy as np
import pytest

from qibolab.instruments.abstract import Instrument
from qibolab.instruments.qblox.cluster import Cluster
from qibolab.instruments.qblox.cluster_qcm_rf import (
    ClusterQCM_RF,
    ClusterQCM_RF_Settings,
)
from qibolab.instruments.qblox.port import (
    ClusterRF_OutputPort,
    ClusterRF_OutputPort_Settings,
)
from qibolab.pulses import DrivePulse, PulseSequence
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from .qblox_fixtures import cluster, connected_cluster, connected_controller, controller

O1_OUTPUT_CHANNEL = "L3-11"
O1_ATTENUATION = 20
O1_LO_FREQUENCY = 5_052_833_073
O1_GAIN = 0.570

O2_OUTPUT_CHANNEL = "L3-12"
O2_ATTENUATION = 20
O2_LO_FREQUENCY = 5_995_371_914
O2_GAIN = 0.655


def get_qcm_rf(controller):
    settings = ClusterQCM_RF_Settings(
        {
            "o1": ClusterRF_OutputPort_Settings(
                channel=O1_OUTPUT_CHANNEL,
                attenuation=O1_ATTENUATION,
                lo_frequency=O1_LO_FREQUENCY,
                gain=O1_GAIN,
            ),
            "o2": ClusterRF_OutputPort_Settings(
                channel=O2_OUTPUT_CHANNEL,
                attenuation=O2_ATTENUATION,
                lo_frequency=O2_LO_FREQUENCY,
                gain=O2_GAIN,
            ),
        }
    )
    for module in controller.modules.values():
        if isinstance(module, ClusterQCM_RF):
            return ClusterQCM_RF(module.name, module.address, settings)
    pytest.skip(f"Skipping qblox ClusterQCM_RF test for {controller.name}.")


@pytest.fixture(scope="module")
def qcm_rf(controller):
    return get_qcm_rf(controller)


@pytest.fixture(scope="module")
def connected_qcm_rf(connected_cluster, connected_controller):
    qcm_rf = get_qcm_rf(connected_controller)
    connected_cluster.connect()
    qcm_rf.connect(connected_cluster.device)
    qcm_rf.setup()
    yield qcm_rf
    qcm_rf.disconnect()
    connected_cluster.disconnect()


def test_ClusterQCM_RF_Settings():
    # Test default value
    qcm_rf_settings = ClusterQCM_RF_Settings()
    for port in ["o1", "o2"]:
        assert port in qcm_rf_settings.ports


def test_instrument_interface(qcm_rf: ClusterQCM_RF):
    # Test compliance with :class:`qibolab.instruments.abstract.Instrument` interface
    for abstract_method in Instrument.__abstractmethods__:
        assert hasattr(qcm_rf, abstract_method)

    for attribute in ["name", "address", "is_connected", "signature", "tmp_folder", "data_folder"]:
        assert hasattr(qcm_rf, attribute)


def test_init(qcm_rf: ClusterQCM_RF):
    assert type(qcm_rf.settings.ports["o1"]) == ClusterRF_OutputPort_Settings
    assert type(qcm_rf.settings.ports["o2"]) == ClusterRF_OutputPort_Settings
    assert qcm_rf.device == None
    for port in ["o1", "o2"]:
        assert port in qcm_rf.ports
    o1_output_port: ClusterRF_OutputPort = qcm_rf.ports["o1"]
    o2_output_port: ClusterRF_OutputPort = qcm_rf.ports["o2"]
    assert o1_output_port.sequencer_number == 0
    assert o2_output_port.sequencer_number == 1


@pytest.mark.qpu
def test_connect(connected_cluster: Cluster, connected_qcm_rf: ClusterQCM_RF):
    cluster = connected_cluster
    qcm_rf = connected_qcm_rf

    cluster.connect()
    qcm_rf.connect(cluster.device)
    assert qcm_rf.is_connected
    assert not qcm_rf is None
    # test configuration after connection
    assert qcm_rf.device.get("out0_offset_path0") == 0
    assert qcm_rf.device.get("out0_offset_path1") == 0
    assert qcm_rf.device.get("out1_offset_path0") == 0
    assert qcm_rf.device.get("out1_offset_path1") == 0

    o1_default_sequencer = qcm_rf.device.sequencers[qcm_rf.DEFAULT_SEQUENCERS["o1"]]
    o2_default_sequencer = qcm_rf.device.sequencers[qcm_rf.DEFAULT_SEQUENCERS["o2"]]
    for default_sequencer in [o1_default_sequencer, o2_default_sequencer]:
        assert default_sequencer.get("cont_mode_en_awg_path0") == False
        assert default_sequencer.get("cont_mode_en_awg_path1") == False
        assert default_sequencer.get("cont_mode_waveform_idx_awg_path0") == 0
        assert default_sequencer.get("cont_mode_waveform_idx_awg_path1") == 0
        assert default_sequencer.get("marker_ovr_en") == True
        assert default_sequencer.get("marker_ovr_value") == 15
        assert default_sequencer.get("mixer_corr_gain_ratio") == 1
        assert default_sequencer.get("mixer_corr_phase_offset_degree") == 0
        assert default_sequencer.get("offset_awg_path0") == 0
        assert default_sequencer.get("offset_awg_path1") == 0
        assert default_sequencer.get("sync_en") == False
        assert default_sequencer.get("upsample_rate_awg_path0") == 0
        assert default_sequencer.get("upsample_rate_awg_path1") == 0

    assert o1_default_sequencer.get("channel_map_path0_out0_en") == True
    assert o1_default_sequencer.get("channel_map_path1_out1_en") == True
    assert o1_default_sequencer.get("channel_map_path0_out2_en") == False
    assert o1_default_sequencer.get("channel_map_path1_out3_en") == False

    assert o2_default_sequencer.get("channel_map_path0_out0_en") == False
    assert o2_default_sequencer.get("channel_map_path1_out1_en") == False
    assert o2_default_sequencer.get("channel_map_path0_out2_en") == True
    assert o2_default_sequencer.get("channel_map_path1_out3_en") == True

    _device_num_sequencers = len(qcm_rf.device.sequencers)
    for s in range(2, _device_num_sequencers):
        assert qcm_rf.device.sequencers[s].get("channel_map_path0_out0_en") == False
        assert qcm_rf.device.sequencers[s].get("channel_map_path1_out1_en") == False
        assert qcm_rf.device.sequencers[s].get("channel_map_path0_out2_en") == False
        assert qcm_rf.device.sequencers[s].get("channel_map_path1_out3_en") == False


@pytest.mark.qpu
def test_setup(connected_cluster: Cluster, connected_qcm_rf: ClusterQCM_RF):
    cluster = connected_cluster
    qcm_rf = connected_qcm_rf

    cluster.connect()
    qcm_rf.connect(cluster.device)
    qcm_rf.setup()

    assert qcm_rf.ports["o1"].channel == O1_OUTPUT_CHANNEL
    assert qcm_rf.device.get("out0_att") == O1_ATTENUATION
    assert qcm_rf.device.get("out0_lo_en") == True
    assert qcm_rf.device.get("out0_lo_freq") == O1_LO_FREQUENCY
    assert qcm_rf.device.get("out0_lo_freq") == O1_LO_FREQUENCY
    import math

    o1_default_sequencer = qcm_rf.device.sequencers[qcm_rf.DEFAULT_SEQUENCERS["o1"]]
    assert math.isclose(o1_default_sequencer.get("gain_awg_path0"), O1_GAIN, rel_tol=1e-4)
    assert math.isclose(o1_default_sequencer.get("gain_awg_path1"), O1_GAIN, rel_tol=1e-4)

    assert o1_default_sequencer.get("mod_en_awg") == True

    assert qcm_rf.ports["o1"].nco_freq == 0
    assert qcm_rf.ports["o1"].nco_phase_offs == 0

    assert qcm_rf.ports["o2"].channel == O2_OUTPUT_CHANNEL
    assert qcm_rf.device.get("out1_att") == O2_ATTENUATION
    assert qcm_rf.device.get("out1_lo_en") == True
    assert qcm_rf.device.get("out1_lo_freq") == O2_LO_FREQUENCY
    assert qcm_rf.device.get("out1_lo_freq") == O2_LO_FREQUENCY
    import math

    o2_default_sequencer = qcm_rf.device.sequencers[qcm_rf.DEFAULT_SEQUENCERS["o2"]]
    assert math.isclose(o2_default_sequencer.get("gain_awg_path0"), O2_GAIN, rel_tol=1e-4)
    assert math.isclose(o2_default_sequencer.get("gain_awg_path1"), O2_GAIN, rel_tol=1e-4)

    assert o2_default_sequencer.get("mod_en_awg") == True

    assert qcm_rf.ports["o2"].nco_freq == 0
    assert qcm_rf.ports["o2"].nco_phase_offs == 0


@pytest.mark.qpu
def test_pulse_sequence(connected_platform, connected_qcm_rf: ClusterQCM_RF):
    ps = PulseSequence()
    ps.add(DrivePulse(0, 200, 1, O1_LO_FREQUENCY - 200e6, np.pi / 2, "Gaussian(5)", O1_OUTPUT_CHANNEL))
    ps.add(DrivePulse(0, 200, 1, O2_LO_FREQUENCY - 200e6, np.pi / 2, "Gaussian(5)", O2_OUTPUT_CHANNEL))

    qubits = connected_platform.qubits
    connected_qcm_rf.ports["o2"].hardware_mod_en = True
    connected_qcm_rf.process_pulse_sequence(qubits, ps, 1000, 1, 10000)
    connected_qcm_rf.upload()
    connected_qcm_rf.play_sequence()

    connected_qcm_rf.ports["o2"].hardware_mod_en = False
    connected_qcm_rf.process_pulse_sequence(qubits, ps, 1000, 1, 10000)
    connected_qcm_rf.upload()
    connected_qcm_rf.play_sequence()


@pytest.mark.qpu
def test_sweepers(connected_platform, connected_qcm_rf: ClusterQCM_RF):
    ps = PulseSequence()
    ps.add(DrivePulse(0, 200, 1, O1_LO_FREQUENCY - 200e6, np.pi / 2, "Gaussian(5)", O1_OUTPUT_CHANNEL))
    ps.add(DrivePulse(0, 200, 1, O2_LO_FREQUENCY - 200e6, np.pi / 2, "Gaussian(5)", O2_OUTPUT_CHANNEL))

    qubits = connected_platform.qubits

    freq_width = 300e6 * 2
    freq_step = freq_width // 100

    delta_frequency_range = np.arange(-freq_width // 2, freq_width // 2, freq_step)
    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=ps.pulses,
        type=SweeperType.OFFSET,
    )

    connected_qcm_rf.process_pulse_sequence(qubits, ps, 1000, 1, 10000, sweepers=[sweeper])
    connected_qcm_rf.upload()
    connected_qcm_rf.play_sequence()

    amplitude_range = np.linspace(0, 1, 50)
    sweeper = Sweeper(
        Parameter.amplitude,
        amplitude_range,
        pulses=ps.pulses,
        type=SweeperType.ABSOLUTE,
    )

    connected_qcm_rf.process_pulse_sequence(qubits, ps, 1000, 1, 10000, sweepers=[sweeper])
    connected_qcm_rf.upload()
    connected_qcm_rf.play_sequence()


@pytest.mark.qpu
def test_start_stop(connected_qcm_rf: ClusterQCM_RF):
    connected_qcm_rf.start()
    connected_qcm_rf.stop()
    # check all sequencers are stopped and all offsets = 0
