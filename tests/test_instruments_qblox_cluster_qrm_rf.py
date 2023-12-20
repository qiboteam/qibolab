import numpy as np
import pytest

from qibolab.instruments.abstract import Instrument
from qibolab.instruments.qblox.cluster import Cluster
from qibolab.instruments.qblox.cluster_qrm_rf import ClusterQRM_RF
from qibolab.instruments.qblox.port import QbloxInputPort, QbloxOutputPort
from qibolab.pulses import DrivePulse, PulseSequence, ReadoutPulse
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from .qblox_fixtures import cluster, connected_cluster, connected_controller, controller

OUTPUT_CHANNEL = "L3-25_a"
INPUT_CHANNEL = "L2-5_a"
ATTENUATION = 38
LO_FREQUENCY = 7_000_000_000
TIME_OF_FLIGHT = 500
ACQUISITION_DURATION = 900


def get_qrm_rf(controller, cluster):
    for module in controller.modules.values():
        if isinstance(module, ClusterQRM_RF):
            return ClusterQRM_RF(module.name, module.address, cluster)
    pytest.skip(f"Skipping qblox ClusterQRM_RF test for {cluster.name}.")


@pytest.fixture(scope="module")
def qrm_rf(controller, cluster):
    return get_qrm_rf(controller, cluster)


@pytest.fixture(scope="module")
def connected_qrm_rf(connected_controller, connected_cluster):
    settings = {
        "o1": {
            "attenuation": ATTENUATION,
            "lo_frequency": LO_FREQUENCY,
        },
        "i1": {
            "acquisition_hold_off": TIME_OF_FLIGHT,
            "acquisition_duration": ACQUISITION_DURATION,
        },
    }
    qrm_rf = get_qrm_rf(connected_controller, connected_cluster)
    qrm_rf.setup(**settings)
    qrm_rf.connect()

    yield qrm_rf
    qrm_rf.disconnect()
    connected_cluster.disconnect()


def test_instrument_interface(qrm_rf: ClusterQRM_RF):
    # Test compliance with :class:`qibolab.instruments.abstract.Instrument` interface
    for abstract_method in Instrument.__abstractmethods__:
        assert hasattr(qrm_rf, abstract_method)

    for attribute in [
        "name",
        "address",
        "is_connected",
        "signature",
        "tmp_folder",
        "data_folder",
    ]:
        assert hasattr(qrm_rf, attribute)


def test_init(qrm_rf: ClusterQRM_RF):
    assert type(qrm_rf._cluster) == Cluster
    assert qrm_rf.device == None


def test_setup(qrm_rf: ClusterQRM_RF):
    settings = {
        "o1": {
            "attenuation": ATTENUATION,
            "lo_frequency": LO_FREQUENCY,
        },
        "i1": {
            "acquisition_hold_off": TIME_OF_FLIGHT,
            "acquisition_duration": ACQUISITION_DURATION,
        },
    }
    qrm_rf.setup(**settings)
    assert type(qrm_rf.ports["o1"]) == QbloxOutputPort
    assert type(qrm_rf.ports["i1"]) == QbloxInputPort
    assert qrm_rf.settings == settings
    output_port: QbloxOutputPort = qrm_rf.ports["o1"]
    assert output_port.sequencer_number == 0
    input_port: QbloxInputPort = qrm_rf.ports["i1"]
    assert input_port.input_sequencer_number == 0
    assert input_port.output_sequencer_number == 0


@pytest.mark.qpu
def test_connect(connected_qrm_rf: ClusterQRM_RF):
    qrm_rf = connected_qrm_rf

    assert qrm_rf.is_connected
    assert not qrm_rf is None
    # test configuration after connection
    assert qrm_rf.device.get("in0_att") == 0
    assert qrm_rf.device.get("out0_offset_path0") == 0
    assert qrm_rf.device.get("out0_offset_path1") == 0
    assert qrm_rf.device.get("scope_acq_avg_mode_en_path0") == True
    assert qrm_rf.device.get("scope_acq_avg_mode_en_path1") == True
    assert (
        qrm_rf.device.get("scope_acq_sequencer_select")
        == qrm_rf.DEFAULT_SEQUENCERS["i1"]
    )
    assert qrm_rf.device.get("scope_acq_trigger_level_path0") == 0
    assert qrm_rf.device.get("scope_acq_trigger_level_path1") == 0
    assert qrm_rf.device.get("scope_acq_trigger_mode_path0") == "sequencer"
    assert qrm_rf.device.get("scope_acq_trigger_mode_path1") == "sequencer"

    default_sequencer = qrm_rf.device.sequencers[qrm_rf.DEFAULT_SEQUENCERS["o1"]]
    assert default_sequencer.get("connect_out0") == "IQ"
    assert default_sequencer.get("connect_acq") == "in0"
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

    _device_num_sequencers = len(qrm_rf.device.sequencers)
    for s in range(1, _device_num_sequencers):
        assert qrm_rf.device.sequencers[s].get("connect_out0") == "off"
        assert qrm_rf.device.sequencers[s].get("connect_acq") == "off"

    assert qrm_rf.device.get("out0_att") == ATTENUATION
    assert qrm_rf.device.get("out0_in0_lo_en") == True
    assert qrm_rf.device.get("out0_in0_lo_freq") == LO_FREQUENCY
    assert qrm_rf.device.get("out0_in0_lo_freq") == LO_FREQUENCY

    default_sequencer = qrm_rf.device.sequencers[qrm_rf.DEFAULT_SEQUENCERS["o1"]]

    assert default_sequencer.get("mod_en_awg") == True

    assert qrm_rf.ports["o1"].nco_freq == 0
    assert qrm_rf.ports["o1"].nco_phase_offs == 0

    assert default_sequencer.get("demod_en_acq") == True

    assert qrm_rf.ports["i1"].acquisition_hold_off == TIME_OF_FLIGHT
    assert qrm_rf.ports["i1"].acquisition_duration == ACQUISITION_DURATION


@pytest.mark.qpu
def test_pulse_sequence(connected_platform, connected_qrm_rf: ClusterQRM_RF):
    ps = PulseSequence()
    for channel in connected_qrm_rf.channel_map:
        ps.add(DrivePulse(0, 200, 1, 6.8e9, np.pi / 2, "Gaussian(5)", channel))
        ps.add(
            ReadoutPulse(
                200, 2000, 1, 7.1e9, np.pi / 2, "Rectangular()", channel, qubit=0
            )
        )
        ps.add(
            ReadoutPulse(
                200, 2000, 1, 7.2e9, np.pi / 2, "Rectangular()", channel, qubit=1
            )
        )
    qubits = connected_platform.qubits
    connected_qrm_rf.ports["i1"].hardware_demod_en = True
    connected_qrm_rf.process_pulse_sequence(qubits, ps, 1000, 1, 10000)
    connected_qrm_rf.upload()
    connected_qrm_rf.play_sequence()
    results = connected_qrm_rf.acquire()
    connected_qrm_rf.ports["i1"].hardware_demod_en = False
    connected_qrm_rf.process_pulse_sequence(qubits, ps, 1000, 1, 10000)
    connected_qrm_rf.upload()
    connected_qrm_rf.play_sequence()
    results = connected_qrm_rf.acquire()


@pytest.mark.qpu
def test_sweepers(connected_platform, connected_qrm_rf: ClusterQRM_RF):
    ps = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    for channel in connected_qrm_rf.channel_map:
        qd_pulses[0] = DrivePulse(
            0, 200, 1, 7e9, np.pi / 2, "Gaussian(5)", channel, qubit=0
        )
        ro_pulses[0] = ReadoutPulse(
            200, 2000, 1, 7.1e9, np.pi / 2, "Rectangular()", channel, qubit=0
        )
        ro_pulses[1] = ReadoutPulse(
            200, 2000, 1, 7.2e9, np.pi / 2, "Rectangular()", channel, qubit=1
        )
        ps.add(qd_pulses[0], ro_pulses[0], ro_pulses[1])

    qubits = connected_platform.qubits

    freq_width = 300e6 * 2
    freq_step = freq_width // 100

    delta_frequency_range = np.arange(-freq_width // 2, freq_width // 2, freq_step)
    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=ro_pulses,
        type=SweeperType.OFFSET,
    )

    connected_qrm_rf.process_pulse_sequence(
        qubits, ps, 1000, 1, 10000, sweepers=[sweeper]
    )
    connected_qrm_rf.upload()
    connected_qrm_rf.play_sequence()
    results = connected_qrm_rf.acquire()

    delta_duration_range = np.arange(0, 140, 1)
    sweeper = Sweeper(
        Parameter.duration,
        delta_duration_range,
        pulses=qd_pulses,
        type=SweeperType.ABSOLUTE,
    )

    connected_qrm_rf.process_pulse_sequence(
        qubits, ps, 1000, 1, 10000, sweepers=[sweeper]
    )
    connected_qrm_rf.upload()
    connected_qrm_rf.play_sequence()
    results = connected_qrm_rf.acquire()


def test_process_acquisition_results():
    pass


@pytest.mark.qpu
def test_start_stop(connected_qrm_rf: ClusterQRM_RF):
    connected_qrm_rf.start()
    connected_qrm_rf.stop()
    # check all sequencers are stopped and all offsets = 0
