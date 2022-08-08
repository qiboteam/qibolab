# -*- coding: utf-8 -*-
import numpy as np
import pytest
import yaml

from qibolab.instruments.qblox import ClusterQCM, ClusterQRM, PulsarQCM, PulsarQRM
from qibolab.paths import qibolab_folder
from qibolab.platforms.multiqubit import MultiqubitPlatform
from qibolab.pulses import Pulse, ReadoutPulse

INSTRUMENTS_LIST = ["ClusterQCM", "ClusterQRM", "PulsarQCM", "PulsarQRM"]
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
    assert instance.signature == f"{name}@{address}"
    assert instance.data_folder == qibolab_folder / "instruments" / "data"


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
        instruments[name].setup(
            **settings["settings"], **settings["instruments"][name]["settings"]
        )

        for parameter in settings["instruments"][name]["settings"]:
            assert (
                getattr(instruments[name], parameter)
                == settings["instruments"][name]["settings"][parameter]
            )


def instrument_set_and_test_parameter_values(instrument, parameter, values):
    for value in values:
        instrument.set_device_parameter(parameter, value)
        assert instrument.device.get(parameter) == value


@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_qublox_set_device_paramter(name):
    if not instruments[name].is_connected:
        pytest.xfail("Instrument not available")
    else:
        instrument_set_and_test_parameter_values(
            instruments[name], "reference_source", ["external", "internal"]
        )
        for sequencer in range(instruments[name].device_num_sequencers):
            for gain in np.arange(0, 1, 0.1):
                instruments[name].set_device_parameter(
                    f"sequencer{sequencer}_gain_awg_path0", gain
                )
                instruments[name].set_device_parameter(
                    f"sequencer{sequencer}_gain_awg_path1", gain
                )
                assert (
                    abs(
                        instruments[name].device.get(
                            f"sequencer{sequencer}_gain_awg_path0"
                        )
                        - gain
                    )
                    < 0.001
                )
                assert (
                    abs(
                        instruments[name].device.get(
                            f"sequencer{sequencer}_gain_awg_path1"
                        )
                        - gain
                    )
                    < 0.001
                )
            instrument_set_and_test_parameter_values(
                instruments[name], f"sequencer{sequencer}_sync_en", [False, True]
            )
            for out in range(0, 2 * instruments[name].device_num_ports):
                instrument_set_and_test_parameter_values(
                    instruments[name],
                    f"sequencer{sequencer}_channel_map_path{out%2}_out{out}_en",
                    [False, True],
                )

        if "QRM" in name:
            instrument_set_and_test_parameter_values(
                instruments[name],
                "scope_acq_trigger_mode_path0",
                ["sequencer", "level"],
            )
            instrument_set_and_test_parameter_values(
                instruments[name],
                "scope_acq_trigger_mode_path1",
                ["sequencer", "level"],
            )
            instrument_set_and_test_parameter_values(
                instruments[name], "scope_acq_avg_mode_en_path0", [False, True]
            )
            instrument_set_and_test_parameter_values(
                instruments[name], "scope_acq_avg_mode_en_path1", [False, True]
            )
            instrument_set_and_test_parameter_values(
                instruments[name],
                "scope_acq_sequencer_select",
                range(instruments[name].device_num_sequencers),
            )

            """   # TODO: add attitional paramter tests
            qrm:
                parameter                                  value
            ------------------------------------------------------------------
            IDN                                         :	None
            in0_gain                                    :	None (dB)
            in1_gain                                    :	None (dB)
            out0_offset                                 :	None (V)
            out1_offset                                 :	None (V)
            reference_source                            :	None
            scope_acq_avg_mode_en_path0                 :	None
            scope_acq_avg_mode_en_path1                 :	None
            scope_acq_sequencer_select                  :	None
            scope_acq_trigger_level_path0               :	None
            scope_acq_trigger_level_path1               :	None
            scope_acq_trigger_mode_path0                :	None
            scope_acq_trigger_mode_path1                :	None
            sequencer0_channel_map_path0_out0_en        :	None
            sequencer0_channel_map_path1_out1_en        :	None
            sequencer0_cont_mode_en_awg_path0           :	None
            sequencer0_cont_mode_en_awg_path1           :	None
            sequencer0_cont_mode_waveform_idx_awg_path0 :	None
            sequencer0_cont_mode_waveform_idx_awg_path1 :	None
            sequencer0_demod_en_acq                     :	None
            sequencer0_discretization_threshold_acq     :	None
            sequencer0_gain_awg_path0                   :	None
            sequencer0_gain_awg_path1                   :	None
            sequencer0_integration_length_acq           :	None
            sequencer0_marker_ovr_en                    :	None
            sequencer0_marker_ovr_value                 :	None
            sequencer0_mixer_corr_gain_ratio            :	None
            sequencer0_mixer_corr_phase_offset_degree   :	None
            sequencer0_mod_en_awg                       :	None
            sequencer0_nco_freq                         :	None (Hz)
            sequencer0_nco_phase_offs                   :	None (Degrees)
            sequencer0_offset_awg_path0                 :	None
            sequencer0_offset_awg_path1                 :	None
            sequencer0_phase_rotation_acq               :	None (Degrees)
            sequencer0_sync_en                          :	None
            sequencer0_upsample_rate_awg_path0          :	None
            sequencer0_upsample_rate_awg_path1          :	None
            sequencer0_waveforms_and_program            :	None

            qcm:
                parameter                                  value
            ----------------------------------------------------------------
            IDN                                         :	None
            out0_offset                                 :	None (V)
            out1_offset                                 :	None (V)
            out2_offset                                 :	None (V)
            out3_offset                                 :	None (V)
            reference_source                            :	None
            sequencer0_channel_map_path0_out0_en        :	None
            sequencer0_channel_map_path0_out2_en        :	None
            sequencer0_channel_map_path1_out1_en        :	None
            sequencer0_channel_map_path1_out3_en        :	None
            sequencer0_cont_mode_en_awg_path0           :	None
            sequencer0_cont_mode_en_awg_path1           :	None
            sequencer0_cont_mode_waveform_idx_awg_path0 :	None
            sequencer0_cont_mode_waveform_idx_awg_path1 :	None
            sequencer0_gain_awg_path0                   :	None
            sequencer0_gain_awg_path1                   :	None
            sequencer0_marker_ovr_en                    :	None
            sequencer0_marker_ovr_value                 :	None
            sequencer0_mixer_corr_gain_ratio            :	None
            sequencer0_mixer_corr_phase_offset_degree   :	None
            sequencer0_mod_en_awg                       :	None
            sequencer0_nco_freq                         :	None (Hz)
            sequencer0_nco_phase_offs                   :	None (Degrees)
            sequencer0_offset_awg_path0                 :	None
            sequencer0_offset_awg_path1                 :	None
            sequencer0_sync_en                          :	None
            sequencer0_upsample_rate_awg_path0          :	None
            sequencer0_upsample_rate_awg_path1          :	None
            sequencer0_waveforms_and_program            :	None
            """

        instruments[name].device.reset()


@pytest.mark.parametrize("name", INSTRUMENTS_LIST)
def test_instruments_process_pulse_sequence_upload_play(name):
    if not instruments[name].is_connected:
        pytest.xfail("Instrument not available")
    else:
        test_runcard = qibolab_folder / "tests" / "test_instruments_qblox.yml"
        with open(test_runcard, "r") as file:
            settings = yaml.safe_load(file)
        instruments[name].setup(
            **settings["settings"], **settings["instruments"][name]["settings"]
        )

        instrument_pulses = {}
        instrument_pulses[name] = {}
        if "QCM" in name:
            for channel in instruments[name].channel_port_map:
                instrument_pulses[name][channel] = [
                    Pulse(0, 200, 1, 10e6, np.pi / 2, "Gaussian(5)", channel)
                ]
            instruments[name].process_pulse_sequence(instrument_pulses[name], nshots=5)
            instruments[name].upload()
            instruments[name].play_sequence()
        if "QRM" in name:
            for channel in instruments[name].channel_port_map:
                instrument_pulses[name][channel] = [
                    Pulse(0, 200, 1, 10e6, np.pi / 2, "Gaussian(5)", channel),
                    ReadoutPulse(
                        200, 2000, 1, 10e6, np.pi / 2, "Rectangular()", channel
                    ),
                ]
            instruments[name].process_pulse_sequence(instrument_pulses[name], nshots=5)
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
