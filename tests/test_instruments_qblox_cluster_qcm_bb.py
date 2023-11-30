import math

import numpy as np
import pytest

from qibolab.instruments.abstract import Instrument
from qibolab.instruments.qblox.cluster_qcm_bb import Cluster, ClusterQCM_BB
from qibolab.instruments.qblox.port import QbloxOutputPort
from qibolab.pulses import FluxPulse, PulseSequence
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from .qblox_fixtures import cluster, connected_cluster, connected_controller, controller

O1_OUTPUT_CHANNEL = "L4-5"
O1_OFFSET = 0.2227

O2_OUTPUT_CHANNEL = "L4-1"
O2_OFFSET = 0.3780

O3_OUTPUT_CHANNEL = "L4-2"
O3_OFFSET = -0.8899

O4_OUTPUT_CHANNEL = "L4-3"
O4_OFFSET = 0.5890


def get_qcm_bb(controller, cluster):
    for module in controller.modules.values():
        if isinstance(module, ClusterQCM_BB):
            return ClusterQCM_BB(module.name, module.address, cluster)
    pytest.skip(f"Skipping qblox ClusterQCM_BB test for {cluster.name}.")


@pytest.fixture(scope="module")
def qcm_bb(controller, cluster):
    return get_qcm_bb(controller, cluster)


@pytest.fixture(scope="module")
def connected_qcm_bb(connected_controller, connected_cluster):
    settings = {
        "o1": {
            "offset": O1_OFFSET,
        },
        "o2": {
            "offset": O2_OFFSET,
        },
        "o3": {
            "offset": O3_OFFSET,
        },
        "o4": {
            "offset": O4_OFFSET,
        },
    }
    qcm_bb = get_qcm_bb(connected_controller, connected_cluster)
    qcm_bb.setup(**settings)
    qcm_bb.connect()
    yield qcm_bb
    qcm_bb.disconnect()
    connected_controller.disconnect()


def test_instrument_interface(qcm_bb: ClusterQCM_BB):
    # Test compliance with :class:`qibolab.instruments.abstract.Instrument` interface
    for abstract_method in Instrument.__abstractmethods__:
        assert hasattr(qcm_bb, abstract_method)

    for attribute in ["name", "address", "is_connected", "signature", "tmp_folder", "data_folder"]:
        assert hasattr(qcm_bb, attribute)


def test_init(qcm_bb: ClusterQCM_BB):
    assert qcm_bb.device == None
    assert type(qcm_bb._cluster) == Cluster


def test_setup(qcm_bb: ClusterQCM_BB):
    settings = {
        "o1": {
            "offset": O1_OFFSET,
        },
        "o2": {
            "offset": O2_OFFSET,
        },
        "o3": {
            "offset": O3_OFFSET,
        },
        "o4": {
            "offset": O4_OFFSET,
        },
    }
    qcm_bb.setup(**settings)
    for idx, port in enumerate(settings):
        assert type(qcm_bb.ports[port]) == QbloxOutputPort
        assert qcm_bb.ports[port].sequencer_number == idx


@pytest.mark.qpu
def test_connect(connected_qcm_bb: ClusterQCM_BB):
    qcm_bb = connected_qcm_bb

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

    assert o1_default_sequencer.get("connect_out0") == "I"

    assert o2_default_sequencer.get("connect_out1") == "Q"

    assert o3_default_sequencer.get("connect_out2") == "I"

    assert o4_default_sequencer.get("connect_out3") == "Q"

    _device_num_sequencers = len(qcm_bb.device.sequencers)
    for s in range(4, _device_num_sequencers):
        assert qcm_bb.device.sequencers[s].get("connect_out0") == "off"
        assert qcm_bb.device.sequencers[s].get("connect_out1") == "off"
        assert qcm_bb.device.sequencers[s].get("connect_out2") == "off"
        assert qcm_bb.device.sequencers[s].get("connect_out3") == "off"

    o1_default_sequencer = qcm_bb.device.sequencers[qcm_bb.DEFAULT_SEQUENCERS["o1"]]
    assert math.isclose(o1_default_sequencer.get("gain_awg_path1"), 1, rel_tol=1e-4)
    assert math.isclose(qcm_bb.device.get("out0_offset"), O1_OFFSET, rel_tol=1e-3)
    assert o1_default_sequencer.get("mod_en_awg") == True
    assert qcm_bb.ports["o1"].nco_freq == 0
    assert qcm_bb.ports["o1"].nco_phase_offs == 0

    o2_default_sequencer = qcm_bb.device.sequencers[qcm_bb.DEFAULT_SEQUENCERS["o2"]]
    assert math.isclose(o2_default_sequencer.get("gain_awg_path1"), 1, rel_tol=1e-4)
    assert math.isclose(qcm_bb.device.get("out1_offset"), O2_OFFSET, rel_tol=1e-3)
    assert o2_default_sequencer.get("mod_en_awg") == True
    assert qcm_bb.ports["o2"].nco_freq == 0
    assert qcm_bb.ports["o2"].nco_phase_offs == 0

    o3_default_sequencer = qcm_bb.device.sequencers[qcm_bb.DEFAULT_SEQUENCERS["o3"]]
    assert math.isclose(o3_default_sequencer.get("gain_awg_path1"), 1, rel_tol=1e-4)
    assert math.isclose(qcm_bb.device.get("out2_offset"), O3_OFFSET, rel_tol=1e-3)
    assert o3_default_sequencer.get("mod_en_awg") == True
    assert qcm_bb.ports["o3"].nco_freq == 0
    assert qcm_bb.ports["o3"].nco_phase_offs == 0

    o4_default_sequencer = qcm_bb.device.sequencers[qcm_bb.DEFAULT_SEQUENCERS["o4"]]
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
