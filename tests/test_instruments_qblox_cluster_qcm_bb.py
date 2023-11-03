import math

import numpy as np
import pytest

from qibolab.instruments.abstract import Instrument
from qibolab.instruments.qblox.cluster import Cluster
from qibolab.instruments.qblox.cluster_qcm_bb import ClusterQCM_BB
from qibolab.pulses import FluxPulse, PulseSequence
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from .qblox_fixtures import cluster, connected_cluster

O1_OUTPUT_CHANNEL = "L4-1"
O1_GAIN = 0.5
O1_OFFSET = 0.2227
O1_QUBIT = 1

O2_OUTPUT_CHANNEL = "L4-2"
O2_GAIN = 0.5
O2_OFFSET = 0.3780
O2_QUBIT = 2

O3_OUTPUT_CHANNEL = "L4-3"
O3_GAIN = 0.5
O3_OFFSET = -0.8899
O3_QUBIT = 3

O4_OUTPUT_CHANNEL = "L4-4"
O4_GAIN = 0.5
O4_OFFSET = 0.5890
O4_QUBIT = 4


# def get_qcm_bb(controller):
#     settings =
#         {
#             "o1": {
#                 "channel": O1_OUTPUT_CHANNEL,
#                 "gain": O1_GAIN,
#                 "offset": O1_OFFSET,
#                 "qubit": O1_QUBIT
#             },
#             "o2": {
#                 "channel": O2_OUTPUT_CHANNEL,
#                 "gain": O2_GAIN,
#                 "offset": O2_OFFSET,
#                 "qubit": O2_QUBIT
#             },
#             "o3": {
#                 "channel": O3_OUTPUT_CHANNEL,
#                 "gain": O3_GAIN,
#                 "offset": O3_OFFSET,
#                 "qubit": O3_QUBIT
#             },
#             "o4": {
#                 "channel": O4_OUTPUT_CHANNEL,
#                 "gain": O4_GAIN,
#                 "offset": O4_OFFSET,
#                 "qubit": O4_QUBIT
#             }
#         }

#     for module in controller.modules.values():
#         if isinstance(module, ClusterQCM_BB):
#             return ClusterQCM_BB(module.name, module.address, settings)
#     pytest.skip(f"Skipping qblox ClusterQCM_BB test for {controller.name}.")


# @pytest.fixture(scope="module")
# def qcm_bb(controller):
#     return get_qcm_bb(controller)


# @pytest.fixture(scope="module")
# def connected_qcm_bb(connected_cluster, connected_controller):
#     qcm_bb = get_qcm_bb(connected_controller)
#     connected_cluster.connect()
#     qcm_bb.connect(connected_cluster.device)
#     qcm_bb.setup()
#     yield qcm_bb
#     qcm_bb.disconnect()
#     connected_cluster.disconnect()


# def test_ClusterQCM_BB_Settings():
#     # Test default value
#     qcm_bb_settings = ClusterQCM_BB_Settings()
#     for port in ["o1", "o2", "o3", "o4"]:
#         assert port in qcm_bb_settings.ports


def test_instrument_interface(qcm_bb: ClusterQCM_BB):
    # Test compliance with :class:`qibolab.instruments.abstract.Instrument` interface
    for abstract_method in Instrument.__abstractmethods__:
        assert hasattr(qcm_bb, abstract_method)

    for attribute in ["name", "address", "is_connected", "signature", "tmp_folder", "data_folder"]:
        assert hasattr(qcm_bb, attribute)


def test_init(qcm_bb: ClusterQCM_BB):
    assert type(qcm_bb.settings.ports["o1"]) == ClusterBB_OutputPort_Settings
    assert type(qcm_bb.settings.ports["o2"]) == ClusterBB_OutputPort_Settings
    assert type(qcm_bb.settings.ports["o3"]) == ClusterBB_OutputPort_Settings
    assert type(qcm_bb.settings.ports["o4"]) == ClusterBB_OutputPort_Settings
    assert qcm_bb.device == None
    for port in ["o1", "o2", "o3", "o4"]:
        assert port in qcm_bb.ports
    o1_output_port: ClusterBB_OutputPort = qcm_bb.ports["o1"]
    o2_output_port: ClusterBB_OutputPort = qcm_bb.ports["o2"]
    o3_output_port: ClusterBB_OutputPort = qcm_bb.ports["o3"]
    o4_output_port: ClusterBB_OutputPort = qcm_bb.ports["o4"]
    assert o1_output_port.sequencer_number == 0
    assert o2_output_port.sequencer_number == 1
    assert o3_output_port.sequencer_number == 2
    assert o4_output_port.sequencer_number == 3


@pytest.mark.qpu
def test_connect(connected_cluster: Cluster, connected_qcm_bb: ClusterQCM_BB):
    cluster = connected_cluster
    qcm_bb = connected_qcm_bb

    cluster.connect()
    qcm_bb.connect(cluster.device)
    assert qcm_bb.is_connected
    assert not qcm_bb is None

    o1_default_sequencer = qcm_bb.device.sequencers[qcm_bb.DEFAULT_SEQUENCERS["o1"]]
    o2_default_sequencer = qcm_bb.device.sequencers[qcm_bb.DEFAULT_SEQUENCERS["o2"]]
    o3_default_sequencer = qcm_bb.device.sequencers[qcm_bb.DEFAULT_SEQUENCERS["o3"]]
    o4_default_sequencer = qcm_bb.device.sequencers[qcm_bb.DEFAULT_SEQUENCERS["o4"]]
    for default_sequencer in [
        o1_default_sequencer,
        o2_default_sequencer,
        o3_default_sequencer,
        o4_default_sequencer,
    ]:
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
    assert o1_default_sequencer.get("channel_map_path1_out1_en") == False
    assert o1_default_sequencer.get("channel_map_path0_out2_en") == False
    assert o1_default_sequencer.get("channel_map_path1_out3_en") == False

    assert o2_default_sequencer.get("channel_map_path0_out0_en") == False
    assert o2_default_sequencer.get("channel_map_path1_out1_en") == True
    assert o2_default_sequencer.get("channel_map_path0_out2_en") == False
    assert o2_default_sequencer.get("channel_map_path1_out3_en") == False

    assert o3_default_sequencer.get("channel_map_path0_out0_en") == False
    assert o3_default_sequencer.get("channel_map_path1_out1_en") == False
    assert o3_default_sequencer.get("channel_map_path0_out2_en") == True
    assert o3_default_sequencer.get("channel_map_path1_out3_en") == False

    assert o4_default_sequencer.get("channel_map_path0_out0_en") == False
    assert o4_default_sequencer.get("channel_map_path1_out1_en") == False
    assert o4_default_sequencer.get("channel_map_path0_out2_en") == False
    assert o4_default_sequencer.get("channel_map_path1_out3_en") == True

    _device_num_sequencers = len(qcm_bb.device.sequencers)
    for s in range(4, _device_num_sequencers):
        assert qcm_bb.device.sequencers[s].get("channel_map_path0_out0_en") == False
        assert qcm_bb.device.sequencers[s].get("channel_map_path1_out1_en") == False
        assert qcm_bb.device.sequencers[s].get("channel_map_path0_out2_en") == False
        assert qcm_bb.device.sequencers[s].get("channel_map_path1_out3_en") == False


@pytest.mark.qpu
def test_setup(connected_qcm_bb: ClusterQCM_BB):
    qcm_bb = connected_qcm_bb
    qcm_bb.setup()

    assert qcm_bb.ports["o1"].channel == O1_OUTPUT_CHANNEL

    o1_default_sequencer = qcm_bb.device.sequencers[qcm_bb.DEFAULT_SEQUENCERS["o1"]]
    assert math.isclose(o1_default_sequencer.get("gain_awg_path0"), O1_GAIN, rel_tol=1e-4)
    assert math.isclose(o1_default_sequencer.get("gain_awg_path1"), 1, rel_tol=1e-4)
    assert math.isclose(qcm_bb.device.get("out0_offset"), O1_OFFSET, rel_tol=1e-3)
    assert o1_default_sequencer.get("mod_en_awg") == True
    assert qcm_bb.ports["o1"].nco_freq == 0
    assert qcm_bb.ports["o1"].nco_phase_offs == 0

    assert qcm_bb.ports["o2"].channel == O2_OUTPUT_CHANNEL

    o2_default_sequencer = qcm_bb.device.sequencers[qcm_bb.DEFAULT_SEQUENCERS["o2"]]
    assert math.isclose(o2_default_sequencer.get("gain_awg_path0"), O2_GAIN, rel_tol=1e-4)
    assert math.isclose(o2_default_sequencer.get("gain_awg_path1"), 1, rel_tol=1e-4)
    assert math.isclose(qcm_bb.device.get("out1_offset"), O2_OFFSET, rel_tol=1e-3)
    assert o2_default_sequencer.get("mod_en_awg") == True
    assert qcm_bb.ports["o2"].nco_freq == 0
    assert qcm_bb.ports["o2"].nco_phase_offs == 0

    assert qcm_bb.ports["o3"].channel == O3_OUTPUT_CHANNEL

    o3_default_sequencer = qcm_bb.device.sequencers[qcm_bb.DEFAULT_SEQUENCERS["o3"]]
    assert math.isclose(o3_default_sequencer.get("gain_awg_path0"), O3_GAIN, rel_tol=1e-4)
    assert math.isclose(o3_default_sequencer.get("gain_awg_path1"), 1, rel_tol=1e-4)
    assert math.isclose(qcm_bb.device.get("out2_offset"), O3_OFFSET, rel_tol=1e-3)
    assert o3_default_sequencer.get("mod_en_awg") == True
    assert qcm_bb.ports["o3"].nco_freq == 0
    assert qcm_bb.ports["o3"].nco_phase_offs == 0

    assert qcm_bb.ports["o4"].channel == O4_OUTPUT_CHANNEL

    o4_default_sequencer = qcm_bb.device.sequencers[qcm_bb.DEFAULT_SEQUENCERS["o4"]]
    assert math.isclose(o4_default_sequencer.get("gain_awg_path0"), O4_GAIN, rel_tol=1e-4)
    assert math.isclose(o4_default_sequencer.get("gain_awg_path1"), 1, rel_tol=1e-4)
    assert math.isclose(qcm_bb.device.get("out3_offset"), O4_OFFSET, rel_tol=1e-3)
    assert o1_default_sequencer.get("mod_en_awg") == True
    assert qcm_bb.ports["o4"].nco_freq == 0
    assert qcm_bb.ports["o4"].nco_phase_offs == 0


@pytest.mark.qpu
def test_pulse_sequence(connected_platform, connected_qcm_bb: ClusterQCM_BB):
    ps = PulseSequence()
    ps.add(FluxPulse(40, 70, 0.5, "Rectangular", O1_OUTPUT_CHANNEL))
    ps.add(FluxPulse(0, 50, 0.3, "Rectangular", O2_OUTPUT_CHANNEL))
    ps.add(FluxPulse(20, 100, 0.02, "Rectangular", O3_OUTPUT_CHANNEL))
    ps.add(FluxPulse(32, 48, 0.4, "Rectangular", O4_OUTPUT_CHANNEL))
    qubits = connected_platform.qubits
    connected_qcm_bb.ports["o2"].hardware_mod_en = True
    connected_qcm_bb.process_pulse_sequence(qubits, ps, 1000, 1, 10000)
    connected_qcm_bb.upload()
    connected_qcm_bb.play_sequence()

    connected_qcm_bb.ports["o2"].hardware_mod_en = False
    connected_qcm_bb.process_pulse_sequence(qubits, ps, 1000, 1, 10000)
    connected_qcm_bb.upload()
    connected_qcm_bb.play_sequence()


@pytest.mark.qpu
def test_sweepers(connected_platform, connected_qcm_bb: ClusterQCM_BB):
    ps = PulseSequence()
    ps.add(FluxPulse(40, 70, 0.5, "Rectangular", O1_OUTPUT_CHANNEL))
    ps.add(FluxPulse(0, 50, 0.3, "Rectangular", O2_OUTPUT_CHANNEL))
    ps.add(FluxPulse(20, 100, 0.02, "Rectangular", O3_OUTPUT_CHANNEL))
    ps.add(FluxPulse(32, 48, 0.4, "Rectangular", O4_OUTPUT_CHANNEL))
    qubits = connected_platform.qubits

    amplitude_range = np.linspace(0, 0.25, 50)
    sweeper = Sweeper(
        Parameter.amplitude,
        amplitude_range,
        pulses=ps.pulses,
        type=SweeperType.OFFSET,
    )

    connected_qcm_bb.process_pulse_sequence(qubits, ps, 1000, 1, 10000, sweepers=[sweeper])
    connected_qcm_bb.upload()
    connected_qcm_bb.play_sequence()


@pytest.mark.qpu
def test_start_stop(connected_qcm_bb: ClusterQCM_BB):
    connected_qcm_bb.start()
    connected_qcm_bb.stop()
    # check all sequencers are stopped and all offsets = 0
