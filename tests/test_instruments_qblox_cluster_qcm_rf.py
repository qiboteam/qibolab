import numpy as np
import pytest

from qibolab.instruments.abstract import Instrument
from qibolab.instruments.qblox.cluster_qcm_rf import ClusterQCM_RF
from qibolab.instruments.qblox.port import QbloxOutputPort
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from .qblox_fixtures import connected_controller, controller

O1_OUTPUT_CHANNEL = "L3-15"
O1_ATTENUATION = 20
O1_LO_FREQUENCY = 5_052_833_073

O2_OUTPUT_CHANNEL = "L3-11"
O2_ATTENUATION = 20
O2_LO_FREQUENCY = 5_995_371_914
SETTINGS = {
    "o1": {
        "attenuation": O1_ATTENUATION,
        "lo_frequency": O1_LO_FREQUENCY,
    },
    "o2": {
        "attenuation": O2_ATTENUATION,
        "lo_frequency": O2_LO_FREQUENCY,
    },
}


def get_qcm_rf(controller):
    for module in controller.modules.values():
        if isinstance(module, ClusterQCM_RF):
            return ClusterQCM_RF(module.name, module.address)


@pytest.fixture(scope="module")
def qcm_rf(controller):
    return get_qcm_rf(controller)


@pytest.fixture(scope="module")
def connected_qcm_rf(connected_controller):
    qcm_rf = get_qcm_rf(connected_controller)
    qcm_rf.setup(**SETTINGS)
    for port in SETTINGS:
        qcm_rf.ports(port)
    qcm_rf.connect(connected_controller.cluster)
    yield qcm_rf

    qcm_rf.disconnect()
    connected_controller.disconnect()


def test_instrument_interface(qcm_rf: ClusterQCM_RF):
    # Test compliance with :class:`qibolab.instruments.abstract.Instrument` interface
    for abstract_method in Instrument.__abstractmethods__:
        assert hasattr(qcm_rf, abstract_method)

    for attribute in [
        "name",
        "address",
        "is_connected",
    ]:
        assert hasattr(qcm_rf, attribute)


def test_init(qcm_rf: ClusterQCM_RF):
    assert qcm_rf.device == None
    assert type(qcm_rf._ports) == dict


def test_setup(qcm_rf: ClusterQCM_RF):
    qcm_rf.setup(**SETTINGS)
    assert qcm_rf.settings == SETTINGS


@pytest.mark.qpu
def test_connect(connected_qcm_rf: ClusterQCM_RF):
    qcm_rf = connected_qcm_rf

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

    assert o1_default_sequencer.get("connect_out0") == "IQ"
    assert o1_default_sequencer.get("connect_out1") == "off"

    assert o2_default_sequencer.get("connect_out1") == "IQ"
    assert o2_default_sequencer.get("connect_out0") == "off"

    _device_num_sequencers = len(qcm_rf.device.sequencers)
    for s in range(2, _device_num_sequencers):
        assert qcm_rf.device.sequencers[s].get("connect_out0") == "off"
        assert qcm_rf.device.sequencers[s].get("connect_out1") == "off"

    assert qcm_rf.device.get("out0_att") == O1_ATTENUATION
    assert qcm_rf.device.get("out0_lo_en") == True
    assert qcm_rf.device.get("out0_lo_freq") == O1_LO_FREQUENCY
    assert qcm_rf.device.get("out0_lo_freq") == O1_LO_FREQUENCY

    o1_default_sequencer = qcm_rf.device.sequencers[qcm_rf.DEFAULT_SEQUENCERS["o1"]]

    assert o1_default_sequencer.get("mod_en_awg") == True

    assert qcm_rf._ports["o1"].nco_freq == 0
    assert qcm_rf._ports["o1"].nco_phase_offs == 0

    assert qcm_rf.device.get("out1_att") == O2_ATTENUATION
    assert qcm_rf.device.get("out1_lo_en") == True
    assert qcm_rf.device.get("out1_lo_freq") == O2_LO_FREQUENCY
    assert qcm_rf.device.get("out1_lo_freq") == O2_LO_FREQUENCY

    o2_default_sequencer = qcm_rf.device.sequencers[qcm_rf.DEFAULT_SEQUENCERS["o2"]]

    assert o2_default_sequencer.get("mod_en_awg") == True

    assert qcm_rf._ports["o2"].nco_freq == 0
    assert qcm_rf._ports["o2"].nco_phase_offs == 0

    for port in qcm_rf.settings:
        assert type(qcm_rf._ports[port]) == QbloxOutputPort
        assert type(qcm_rf._sequencers[port]) == list
    o1_output_port: QbloxOutputPort = qcm_rf._ports["o1"]
    o2_output_port: QbloxOutputPort = qcm_rf._ports["o2"]
    assert o1_output_port.sequencer_number == 0
    assert o2_output_port.sequencer_number == 1


@pytest.mark.qpu
def test_pulse_sequence(connected_platform, connected_qcm_rf: ClusterQCM_RF):
    ps = PulseSequence()
    ps.append(
        DrivePulse(
            0,
            200,
            1,
            O1_LO_FREQUENCY - 200e6,
            np.pi / 2,
            "Gaussian(5)",
            O1_OUTPUT_CHANNEL,
        )
    )
    ps.append(
        DrivePulse(
            0,
            200,
            1,
            O2_LO_FREQUENCY - 200e6,
            np.pi / 2,
            "Gaussian(5)",
            O2_OUTPUT_CHANNEL,
        )
    )

    qubits = connected_platform.qubits
    connected_qcm_rf._ports["o2"].hardware_mod_en = True
    connected_qcm_rf.process_pulse_sequence(qubits, ps, 1000, 1, 10000)
    connected_qcm_rf.upload()
    connected_qcm_rf.play_sequence()

    connected_qcm_rf._ports["o2"].hardware_mod_en = False
    connected_qcm_rf.process_pulse_sequence(qubits, ps, 1000, 1, 10000)
    connected_qcm_rf.upload()
    connected_qcm_rf.play_sequence()


@pytest.mark.qpu
def test_sweepers(connected_platform, connected_qcm_rf: ClusterQCM_RF):
    ps = PulseSequence()
    ps.append(
        DrivePulse(
            0,
            200,
            1,
            O1_LO_FREQUENCY - 200e6,
            np.pi / 2,
            "Gaussian(5)",
            O1_OUTPUT_CHANNEL,
        )
    )
    ps.append(
        DrivePulse(
            0,
            200,
            1,
            O2_LO_FREQUENCY - 200e6,
            np.pi / 2,
            "Gaussian(5)",
            O2_OUTPUT_CHANNEL,
        )
    )

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

    connected_qcm_rf.process_pulse_sequence(
        qubits, ps, 1000, 1, 10000, sweepers=[sweeper]
    )
    connected_qcm_rf.upload()
    connected_qcm_rf.play_sequence()

    amplitude_range = np.linspace(0, 1, 50)
    sweeper = Sweeper(
        Parameter.amplitude,
        amplitude_range,
        pulses=ps.pulses,
        type=SweeperType.ABSOLUTE,
    )

    connected_qcm_rf.process_pulse_sequence(
        qubits, ps, 1000, 1, 10000, sweepers=[sweeper]
    )
    connected_qcm_rf.upload()
    connected_qcm_rf.play_sequence()
