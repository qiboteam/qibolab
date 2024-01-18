import math

import numpy as np
import pytest

from qibolab.instruments.abstract import Instrument
from qibolab.instruments.qblox.cluster_qcm_bb import ClusterQCM_BB
from qibolab.instruments.qblox.port import QbloxOutputPort
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from .qblox_fixtures import connected_controller, controller

O1_OUTPUT_CHANNEL = "L4-5"
O2_OUTPUT_CHANNEL = "L4-1"
O3_OUTPUT_CHANNEL = "L4-2"
O4_OUTPUT_CHANNEL = "L4-3"

PORT_SETTINGS = ["o1", "o2", "o3", "o4"]


def get_qcm_bb(controller):
    for module in controller.modules.values():
        if isinstance(module, ClusterQCM_BB):
            return ClusterQCM_BB(module.name, module.address)


@pytest.fixture(scope="module")
def qcm_bb(controller):
    return get_qcm_bb(controller)


@pytest.fixture(scope="module")
def connected_qcm_bb(connected_controller):
    qcm_bb = get_qcm_bb(connected_controller)
    for port in PORT_SETTINGS:
        qcm_bb.ports(port)
    qcm_bb.connect(connected_controller.cluster)
    yield qcm_bb
    qcm_bb.disconnect()
    connected_controller.disconnect()


def test_instrument_interface(qcm_bb: ClusterQCM_BB):
    # Test compliance with :class:`qibolab.instruments.abstract.Instrument` interface
    for abstract_method in Instrument.__abstractmethods__:
        assert hasattr(qcm_bb, abstract_method)

    for attribute in [
        "name",
        "address",
        "is_connected",
    ]:
        assert hasattr(qcm_bb, attribute)


def test_init(qcm_bb: ClusterQCM_BB):
    assert qcm_bb.device == None


def test_setup(qcm_bb: ClusterQCM_BB):
    qcm_bb.setup()


@pytest.mark.qpu
def test_connect(connected_qcm_bb: ClusterQCM_BB):
    qcm_bb = connected_qcm_bb

    assert qcm_bb.is_connected
    assert not qcm_bb is None
    for idx, port in enumerate(qcm_bb._ports):
        assert type(qcm_bb._ports[port]) == QbloxOutputPort
        assert qcm_bb._ports[port].sequencer_number == idx

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
    assert o1_default_sequencer.get("mod_en_awg") == True
    assert qcm_bb._ports["o1"].nco_freq == 0
    assert qcm_bb._ports["o1"].nco_phase_offs == 0

    o2_default_sequencer = qcm_bb.device.sequencers[qcm_bb.DEFAULT_SEQUENCERS["o2"]]
    assert math.isclose(o2_default_sequencer.get("gain_awg_path1"), 1, rel_tol=1e-4)
    assert o2_default_sequencer.get("mod_en_awg") == True
    assert qcm_bb._ports["o2"].nco_freq == 0
    assert qcm_bb._ports["o2"].nco_phase_offs == 0

    o3_default_sequencer = qcm_bb.device.sequencers[qcm_bb.DEFAULT_SEQUENCERS["o3"]]
    assert math.isclose(o3_default_sequencer.get("gain_awg_path1"), 1, rel_tol=1e-4)
    assert o3_default_sequencer.get("mod_en_awg") == True
    assert qcm_bb._ports["o3"].nco_freq == 0
    assert qcm_bb._ports["o3"].nco_phase_offs == 0

    o4_default_sequencer = qcm_bb.device.sequencers[qcm_bb.DEFAULT_SEQUENCERS["o4"]]
    assert math.isclose(o4_default_sequencer.get("gain_awg_path1"), 1, rel_tol=1e-4)
    assert o1_default_sequencer.get("mod_en_awg") == True
    assert qcm_bb._ports["o4"].nco_freq == 0
    assert qcm_bb._ports["o4"].nco_phase_offs == 0


@pytest.mark.qpu
def test_pulse_sequence(connected_platform, connected_qcm_bb: ClusterQCM_BB):
    ps = PulseSequence()
    ps.append(FluxPulse(40, 70, 0.5, "Rectangular", O1_OUTPUT_CHANNEL))
    ps.append(FluxPulse(0, 50, 0.3, "Rectangular", O2_OUTPUT_CHANNEL))
    ps.append(FluxPulse(20, 100, 0.02, "Rectangular", O3_OUTPUT_CHANNEL))
    ps.append(FluxPulse(32, 48, 0.4, "Rectangular", O4_OUTPUT_CHANNEL))
    qubits = connected_platform.qubits
    connected_qcm_bb._ports["o2"].hardware_mod_en = True
    connected_qcm_bb.process_pulse_sequence(qubits, ps, 1000, 1, 10000)
    connected_qcm_bb.upload()
    connected_qcm_bb.play_sequence()

    connected_qcm_bb._ports["o2"].hardware_mod_en = False
    connected_qcm_bb.process_pulse_sequence(qubits, ps, 1000, 1, 10000)
    connected_qcm_bb.upload()
    connected_qcm_bb.play_sequence()


@pytest.mark.qpu
def test_sweepers(connected_platform, connected_qcm_bb: ClusterQCM_BB):
    ps = PulseSequence()
    ps.append(FluxPulse(40, 70, 0.5, "Rectangular", O1_OUTPUT_CHANNEL))
    ps.append(FluxPulse(0, 50, 0.3, "Rectangular", O2_OUTPUT_CHANNEL))
    ps.append(FluxPulse(20, 100, 0.02, "Rectangular", O3_OUTPUT_CHANNEL))
    ps.append(FluxPulse(32, 48, 0.4, "Rectangular", O4_OUTPUT_CHANNEL))
    qubits = connected_platform.qubits

    amplitude_range = np.linspace(0, 0.25, 50)
    sweeper = Sweeper(
        Parameter.amplitude,
        amplitude_range,
        pulses=ps.pulses,
        type=SweeperType.OFFSET,
    )

    connected_qcm_bb.process_pulse_sequence(
        qubits, ps, 1000, 1, 10000, sweepers=[sweeper]
    )
    connected_qcm_bb.upload()
    connected_qcm_bb.play_sequence()
