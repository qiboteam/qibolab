# -*- coding: utf-8 -*-
import numpy as np
import pytest
import yaml

from qibolab.instruments.qblox import Cluster, ClusterQCM_RF, ClusterQRM_RF
from qibolab.paths import qibolab_folder, user_folder
from qibolab.platforms.multiqubit import MultiqubitPlatform
from qibolab.pulses import Pulse, PulseSequence, ReadoutPulse

INSTRUMENTS_LIST = ["Cluster", "ClusterQRM_RF", "ClusterQCM_RF"]
instruments = {}


@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qublox_init(name):
    test_runcard = qibolab_folder / "tests" / "test_instruments_qblox.yml"
    with open(test_runcard, "r") as file:
        settings = yaml.safe_load(file)

    # Instantiate instrument
    lib = settings["instruments"][name]["lib"]
    i_class = settings["instruments"][name]["class"]
    address = settings["instruments"][name]["address"]
    from importlib import import_module

    InstrumentClass = getattr(import_module(f"qibolab.instruments.{lib}"), i_class)
    instance = InstrumentClass(name, address)
    instruments[name] = instance
    assert instance.name == name
    assert instance.address == address
    assert instance.is_connected == False
    assert instance.device == None
    assert instance.signature == f"{i_class}@{address}"
    assert instance.data_folder == user_folder / "instruments" / "data" / instance.tmp_folder.name.split("/")[-1]


@pytest.mark.xfail
@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qublox_connect(name):
    instruments[name].connect()


@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qublox_setup(name):
    if not instruments[name].is_connected:
        pytest.xfail("Instrument not available")
    else:
        test_runcard = qibolab_folder / "tests" / "test_instruments_qblox.yml"
        with open(test_runcard, "r") as file:
            settings = yaml.safe_load(file)
        instruments[name].setup(**settings["settings"], **settings["instruments"][name]["settings"])

        for parameter in settings["instruments"][name]["settings"]:
            if parameter == "ports":
                for port in settings["instruments"][name]["settings"]["ports"]:
                    for sub_parameter in settings["instruments"][name]["settings"]["ports"][port]:
                        # assert getattr(instruments[name].ports[port], sub_parameter) == settings["instruments"][name]["settings"]["ports"][port][sub_parameter]
                        np.testing.assert_allclose(
                            getattr(instruments[name].ports[port], sub_parameter),
                            settings["instruments"][name]["settings"]["ports"][port][sub_parameter],
                            atol=1e-4,
                        )
            else:
                assert getattr(instruments[name], parameter) == settings["instruments"][name]["settings"][parameter]


def instrument_test_property_wrapper(
    origin_object, origin_attribute, destination_object, *destination_parameters, values
):
    for value in values:
        setattr(origin_object, origin_attribute, value)
        for destination_parameter in destination_parameters:
            assert (destination_object.get(destination_parameter) == value) or (
                np.testing.assert_allclose(destination_object.get(destination_parameter), value, rtol=1e-1) == None
            )


@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qublox_set_property_wrappers(name):
    if not instruments[name].is_connected:
        pytest.xfail("Instrument not available")
    else:
        instrument = instruments[name]
        device = instruments[name].device

        if name == "Cluster":
            instrument_test_property_wrapper(
                instrument, "reference_clock_source", device, "reference_source", values=["external", "internal"]
            )

        if name == "ClusterQRM_RF":
            port = instruments[name].ports["o1"]
            sequencer = device.sequencers[instrument.DEFAULT_SEQUENCERS["o1"]]

            instrument_test_property_wrapper(port, "attenuation", device, "out0_att", values=np.arange(0, 60 + 2, 2))
            instrument_test_property_wrapper(port, "lo_enabled", device, "out0_in0_lo_en", values=[True, False])
            instrument_test_property_wrapper(
                port, "lo_frequency", device, "out0_in0_lo_freq", values=np.linspace(2e9, 18e9, 20)
            )
            instrument_test_property_wrapper(
                port, "gain", sequencer, "gain_awg_path0", "gain_awg_path1", values=np.linspace(-1, 1, 20)
            )
            instrument_test_property_wrapper(port, "hardware_mod_en", sequencer, "mod_en_awg", values=[True, False])
            instrument_test_property_wrapper(
                port, "nco_freq", sequencer, "nco_freq", values=np.linspace(-300e6, 300e6, 20)
            )
            instrument_test_property_wrapper(
                port, "nco_phase_offs", sequencer, "nco_phase_offs", values=np.linspace(0, 359, 20)
            )

            port = instruments[name].ports["i1"]
            sequencer = device.sequencers[instrument.DEFAULT_SEQUENCERS["i1"]]

            instrument_test_property_wrapper(port, "hardware_demod_en", sequencer, "demod_en_acq", values=[True, False])

            instrument_test_property_wrapper(
                instrument,
                "acquisition_duration",
                sequencer,
                "integration_length_acq",
                values=np.arange(4, 16777212 + 4, 729444),
            )

            instrument_test_property_wrapper(
                instrument,
                "discretization_threshold_acq",
                sequencer,
                "discretization_threshold_acq",
                values=np.linspace(-16777212.0, 16777212.0, 20),
            )

            instrument_test_property_wrapper(
                instrument, "phase_rotation_acq", sequencer, "phase_rotation_acq", values=np.linspace(0, 359, 20)
            )

        if name == "ClusterQCM_RF":
            port = instruments[name].ports["o1"]
            sequencer = device.sequencers[instrument.DEFAULT_SEQUENCERS["o1"]]

            instrument_test_property_wrapper(port, "attenuation", device, "out0_att", values=np.arange(0, 60 + 2, 2))
            instrument_test_property_wrapper(port, "lo_enabled", device, "out0_lo_en", values=[True, False])
            instrument_test_property_wrapper(
                port, "lo_frequency", device, "out0_lo_freq", values=np.linspace(2e9, 18e9, 20)
            )
            instrument_test_property_wrapper(
                port, "gain", sequencer, "gain_awg_path0", "gain_awg_path1", values=np.linspace(-1, 1, 20)
            )
            instrument_test_property_wrapper(port, "hardware_mod_en", sequencer, "mod_en_awg", values=[True, False])
            instrument_test_property_wrapper(
                port, "nco_freq", sequencer, "nco_freq", values=np.linspace(-300e6, 300e6, 20)
            )
            instrument_test_property_wrapper(
                port, "nco_phase_offs", sequencer, "nco_phase_offs", values=np.linspace(0, 359, 20)
            )

            port = instruments[name].ports["o2"]
            sequencer = device.sequencers[instrument.DEFAULT_SEQUENCERS["o2"]]

            instrument_test_property_wrapper(port, "attenuation", device, "out1_att", values=np.arange(0, 60 + 2, 2))
            instrument_test_property_wrapper(port, "lo_enabled", device, "out1_lo_en", values=[True, False])
            instrument_test_property_wrapper(
                port, "lo_frequency", device, "out1_lo_freq", values=np.linspace(2e9, 18e9, 20)
            )
            instrument_test_property_wrapper(
                port, "gain", sequencer, "gain_awg_path0", "gain_awg_path1", values=np.linspace(-1, 1, 20)
            )
            instrument_test_property_wrapper(port, "hardware_mod_en", sequencer, "mod_en_awg", values=[True, False])
            instrument_test_property_wrapper(
                port, "nco_freq", sequencer, "nco_freq", values=np.linspace(-300e6, 300e6, 20)
            )
            instrument_test_property_wrapper(
                port, "nco_phase_offs", sequencer, "nco_phase_offs", values=np.linspace(0, 359, 20)
            )


def instrument_set_and_test_parameter_values(instrument, target, parameter, values):
    for value in values:
        instrument._set_device_parameter(target, parameter, value)
        np.testing.assert_allclose(target.get(parameter), value)


@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qublox_set_device_paramters(name):
    pass
    """   # TODO: add attitional paramter tests
    qrm
        platform.instruments['qrm_rf'].device.print_readable_snapshot(update=True)
        cluster_module16:
            parameter                    value
        --------------------------------------------------------------------------------
        in0_att                       :	0 (dB)
        out0_att                      :	34 (dB)
        out0_in0_lo_en                :	True
        out0_in0_lo_freq              :	7537724144 (Hz)
        out0_offset_path0             :	34 (mV)
        out0_offset_path1             :	0 (mV)
        present                       :	True
        scope_acq_avg_mode_en_path0   :	True
        scope_acq_avg_mode_en_path1   :	True
        scope_acq_sequencer_select    :	0
        scope_acq_trigger_level_path0 :	0
        scope_acq_trigger_level_path1 :	0
        scope_acq_trigger_mode_path0  :	sequencer
        scope_acq_trigger_mode_path1  :	sequencer
        cluster_module16_sequencer0:
            parameter                       value
        --------------------------------------------------------------------------------
        channel_map_path0_out0_en        :	True
        channel_map_path1_out1_en        :	True
        cont_mode_en_awg_path0           :	False
        cont_mode_en_awg_path1           :	False
        cont_mode_waveform_idx_awg_path0 :	0
        cont_mode_waveform_idx_awg_path1 :	0
        demod_en_acq                     :	False
        discretization_threshold_acq     :	0
        gain_awg_path0                   :	1
        gain_awg_path1                   :	1
        integration_length_acq           :	2000
        marker_ovr_en                    :	True
        marker_ovr_value                 :	15
        mixer_corr_gain_ratio            :	1
        mixer_corr_phase_offset_degree   :	-0
        mod_en_awg                       :	False
        nco_freq                         :	0 (Hz)
        nco_phase_offs                   :	0 (Degrees)
        offset_awg_path0                 :	0
        offset_awg_path1                 :	0
        phase_rotation_acq               :	0 (Degrees)
        sequence                         :	/nfs/users/alvaro.orgaz/qibolab/src/qibola...
        sync_en                          :	True
        upsample_rate_awg_path0          :	0
        upsample_rate_awg_path1          :	0

    qcm:
        platform.instruments['qcm_rf2'].device.print_readable_snapshot(update=True)
        cluster_module12:
            parameter        value
        --------------------------------------------------------------------------------
        out0_att          :	24 (dB)
        out0_lo_en        :	True
        out0_lo_freq      :	5325473000 (Hz)
        out0_offset_path0 :	24 (mV)
        out0_offset_path1 :	24 (mV)
        out1_att          :	24 (dB)
        out1_lo_en        :	True
        out1_lo_freq      :	6212286000 (Hz)
        out1_offset_path0 :	0 (mV)
        out1_offset_path1 :	0 (mV)
        present           :	True
        cluster_module12_sequencer0:
            parameter                       value
        --------------------------------------------------------------------------------
        channel_map_path0_out0_en        :	True
        channel_map_path0_out2_en        :	False
        channel_map_path1_out1_en        :	True
        channel_map_path1_out3_en        :	False
        cont_mode_en_awg_path0           :	False
        cont_mode_en_awg_path1           :	False
        cont_mode_waveform_idx_awg_path0 :	0
        cont_mode_waveform_idx_awg_path1 :	0
        gain_awg_path0                   :	0.33998
        gain_awg_path1                   :	0.33998
        marker_ovr_en                    :	True
        marker_ovr_value                 :	15
        mixer_corr_gain_ratio            :	1
        mixer_corr_phase_offset_degree   :	-0
        mod_en_awg                       :	False
        nco_freq                         :	-2e+08 (Hz)
        nco_phase_offs                   :	0 (Degrees)
        offset_awg_path0                 :	0
        offset_awg_path1                 :	0
        sequence                         :	/nfs/users/alvaro.orgaz/qibolab/src/qibola...
        sync_en                          :	True
        upsample_rate_awg_path0          :	0
        upsample_rate_awg_path1          :	0
    """


@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_process_pulse_sequence_upload_play(name):
    if not instruments[name].is_connected:
        pytest.xfail("Instrument not available")
    else:
        test_runcard = qibolab_folder / "tests" / "test_instruments_qblox.yml"
        with open(test_runcard, "r") as file:
            settings = yaml.safe_load(file)
        instruments[name].setup(**settings["settings"], **settings["instruments"][name]["settings"])
        repetition_duration = settings["settings"]["repetition_duration"]

        instrument_pulses = {}
        instrument_pulses[name] = PulseSequence()
        if "QCM" in name:
            for channel in instruments[name].channel_port_map:
                instrument_pulses[name].add(Pulse(0, 200, 1, 10e6, np.pi / 2, "Gaussian(5)", channel))
            instruments[name].process_pulse_sequence(
                instrument_pulses[name], nshots=5, repetition_duration=repetition_duration
            )
            instruments[name].upload()
            instruments[name].play_sequence()
        if "QRM" in name:
            channel = instruments[name]._port_channel_map["o1"]
            instrument_pulses[name].add(
                Pulse(0, 200, 1, 10e6, np.pi / 2, "Gaussian(5)", channel),
                ReadoutPulse(200, 2000, 1, 10e6, np.pi / 2, "Rectangular()", channel),
            )
            instruments[name].device.sequencers[0].sync_en(
                False
            )  # TODO: Check why this is necessary here and not when playing a PS of only one readout pulse
            instruments[name].process_pulse_sequence(
                instrument_pulses[name], nshots=5, repetition_duration=repetition_duration
            )
            instruments[name].upload()
            acquisition_results = instruments[name].play_sequence_and_acquire()


@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qublox_start_stop_disconnect(name):
    if not instruments[name].is_connected:
        pytest.xfail("Instrument not available")
    else:
        instruments[name].start()
        instruments[name].stop()
        instruments[name].disconnect()
        assert instruments[name].is_connected == False
