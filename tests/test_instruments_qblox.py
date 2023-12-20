"""Qblox instruments driver.

Supports the following Instruments:
    Cluster
    Cluster QRM-RF
    Cluster QCM-RF
    Cluster QCM
Compatible with qblox-instruments driver 0.9.0 (28/2/2023).
It supports:
    - multiplexed readout of up to 6 qubits
    - hardware modulation, demodulation, and classification
    - software modulation, with support for arbitrary pulses
    - software demodulation
    - binned acquisition
    - real-time sweepers of
        - pulse frequency (requires hardware modulation)
        - pulse relative phase (requires hardware modulation)
        - pulse amplitude
        - pulse start
        - pulse duration
        - port gain
        - port offset
    - multiple readouts for the same qubit (sequence unrolling)
    - max iq pulse length 8_192ns
    - waveforms cache, uses additional free sequencers if the memory of one sequencer (16384) is exhausted
    - instrument parameters cache
    - safe disconnection of offsets on termination
"""


# from .conftest import load_from_platform

# INSTRUMENTS_LIST = ["Cluster", "ClusterQRM_RF", "ClusterQCM_RF"]

# instruments = {}
# instruments_settings = {}


# @pytest.mark.qpu
# @pytest.mark.parametrize("name", INSTRUMENTS_LIST)
# def test_instruments_qublox_init(platform_name, name):
#     platform = create_platform(platform_name)
#     settings = platform.settings
#     # Instantiate instrument
#     instance, instr_settings = load_from_platform(create_platform(platform_name), name)
#     instruments[name] = instance
#     instruments_settings[name] = instr_settings
#     assert instance.name == name
#     assert instance.is_connected == False
#     assert instance.device == None
#     assert instance.data_folder == INSTRUMENTS_DATA_FOLDER / instance.tmp_folder.name.split("/")[-1]


# @pytest.mark.qpu
# @pytest.mark.parametrize("name", INSTRUMENTS_LIST)
# def test_instruments_qublox_connect(name):
#     instruments[name].connect()


# @pytest.mark.qpu
# @pytest.mark.parametrize("name", INSTRUMENTS_LIST)
# def test_instruments_qublox_setup(platform_name, name):
#     settings = create_platform(platform_name).settings
#     instruments[name].setup(**settings["settings"], **instruments_settings[name])
#     for parameter in instruments_settings[name]:
#         if parameter == "ports":
#             for port in instruments_settings[name]["ports"]:
#                 for sub_parameter in instruments_settings[name]["ports"][port]:
#                     # assert getattr(instruments[name].ports[port], sub_parameter) == settings["instruments"][name]["settings"]["ports"][port][sub_parameter]
#                     np.testing.assert_allclose(
#                         getattr(instruments[name].ports[port], sub_parameter),
#                         instruments_settings[name]["ports"][port][sub_parameter],
#                         atol=1e-4,
#                     )
#         else:
#             assert getattr(instruments[name], parameter) == instruments_settings[name][parameter]


# def instrument_test_property_wrapper(
#     origin_object, origin_attribute, destination_object, *destination_parameters, values
# ):
#     for value in values:
#         setattr(origin_object, origin_attribute, value)
#         for destination_parameter in destination_parameters:
#             assert (destination_object.get(destination_parameter) == value) or (
#                 np.testing.assert_allclose(destination_object.get(destination_parameter), value, rtol=1e-1) == None
#             )


# @pytest.mark.qpu
# @pytest.mark.parametrize("name", INSTRUMENTS_LIST)
# def test_instruments_qublox_set_property_wrappers(name):
#     instrument = instruments[name]
#     device = instruments[name].device
#     if instrument.__class__.__name__ == "Cluster":
#         instrument_test_property_wrapper(
#             instrument, "reference_clock_source", device, "reference_source", values=["external", "internal"]
#         )
#     if instrument.__class__.__name__ == "ClusterQRM_RF":
#         port = instruments[name].ports["o1"]
#         sequencer = device.sequencers[instrument.DEFAULT_SEQUENCERS["o1"]]
#         instrument_test_property_wrapper(port, "attenuation", device, "out0_att", values=np.arange(0, 60 + 2, 2))
#         instrument_test_property_wrapper(port, "lo_enabled", device, "out0_in0_lo_en", values=[True, False])
#         instrument_test_property_wrapper(
#             port, "lo_frequency", device, "out0_in0_lo_freq", values=np.linspace(2e9, 18e9, 20)
#         )
#         instrument_test_property_wrapper(
#             port, "gain", sequencer, "gain_awg_path0", "gain_awg_path1", values=np.linspace(-1, 1, 20)
#         )
#         instrument_test_property_wrapper(port, "hardware_mod_en", sequencer, "mod_en_awg", values=[True, False])
#         instrument_test_property_wrapper(port, "nco_freq", sequencer, "nco_freq", values=np.linspace(-500e6, 500e6, 20))
#         instrument_test_property_wrapper(
#             port, "nco_phase_offs", sequencer, "nco_phase_offs", values=np.linspace(0, 359, 20)
#         )
#         port = instruments[name].ports["i1"]
#         sequencer = device.sequencers[instrument.DEFAULT_SEQUENCERS["i1"]]
#         instrument_test_property_wrapper(port, "hardware_demod_en", sequencer, "demod_en_acq", values=[True, False])
#         instrument_test_property_wrapper(
#             instrument,
#             "acquisition_duration",
#             sequencer,
#             "integration_length_acq",
#             values=np.arange(4, 16777212 + 4, 729444),
#         )
#         # FIXME: I don't know why this is failing
#         instrument_test_property_wrapper(
#             instrument,
#             "thresholded_acq_threshold",
#             sequencer,
#             "thresholded_acq_threshold",
#             # values=np.linspace(-16777212.0, 16777212.0, 20),
#             values=np.zeros(1),
#         )
#         instrument_test_property_wrapper(
#             instrument,
#             "thresholded_acq_rotation",
#             sequencer,
#             "thresholded_acq_rotation",
#             values=np.zeros(1),
#             # values=np.linspace(0, 359, 20)
#         )
#     if instrument.__class__.__name__ == "ClusterQCM_RF":
#         port = instruments[name].ports["o1"]
#         sequencer = device.sequencers[instrument.DEFAULT_SEQUENCERS["o1"]]
#         instrument_test_property_wrapper(port, "attenuation", device, "out0_att", values=np.arange(0, 60 + 2, 2))
#         instrument_test_property_wrapper(port, "lo_enabled", device, "out0_lo_en", values=[True, False])
#         instrument_test_property_wrapper(
#             port, "lo_frequency", device, "out0_lo_freq", values=np.linspace(2e9, 18e9, 20)
#         )
#         instrument_test_property_wrapper(
#             port, "gain", sequencer, "gain_awg_path0", "gain_awg_path1", values=np.linspace(-1, 1, 20)
#         )
#         instrument_test_property_wrapper(port, "hardware_mod_en", sequencer, "mod_en_awg", values=[True, False])
#         instrument_test_property_wrapper(port, "nco_freq", sequencer, "nco_freq", values=np.linspace(-500e6, 500e6, 20))
#         instrument_test_property_wrapper(
#             port, "nco_phase_offs", sequencer, "nco_phase_offs", values=np.linspace(0, 359, 20)
#         )
#         port = instruments[name].ports["o2"]
#         sequencer = device.sequencers[instrument.DEFAULT_SEQUENCERS["o2"]]
#         instrument_test_property_wrapper(port, "attenuation", device, "out1_att", values=np.arange(0, 60 + 2, 2))
#         instrument_test_property_wrapper(port, "lo_enabled", device, "out1_lo_en", values=[True, False])
#         instrument_test_property_wrapper(
#             port, "lo_frequency", device, "out1_lo_freq", values=np.linspace(2e9, 18e9, 20)
#         )
#         instrument_test_property_wrapper(
#             port, "gain", sequencer, "gain_awg_path0", "gain_awg_path1", values=np.linspace(-1, 1, 20)
#         )
#         instrument_test_property_wrapper(port, "hardware_mod_en", sequencer, "mod_en_awg", values=[True, False])
#         instrument_test_property_wrapper(port, "nco_freq", sequencer, "nco_freq", values=np.linspace(-500e6, 500e6, 20))
#         instrument_test_property_wrapper(
#             port, "nco_phase_offs", sequencer, "nco_phase_offs", values=np.linspace(0, 359, 20)
#         )


# def instrument_set_and_test_parameter_values(instrument, target, parameter, values):
#     for value in values:
#         instrument._set_device_parameter(target, parameter, value)
#         np.testing.assert_allclose(target.get(parameter), value)


# @pytest.mark.parametrize("name", INSTRUMENTS_LIST)
# def test_instruments_qublox_set_device_paramters(name):
#     """  # TODO: add attitional paramter tests
#     qrm
#         platform.instruments['qrm_rf'].device.print_readable_snapshot(update=True)
#         cluster_module16:
#             parameter                    value
#         --------------------------------------------------------------------------------
#         in0_att                       :	0 (dB)
#         out0_att                      :	34 (dB)
#         out0_in0_lo_en                :	True
#         out0_in0_lo_freq              :	7537724144 (Hz)
#         out0_offset_path0             :	34 (mV)
#         out0_offset_path1             :	0 (mV)
#         present                       :	True
#         scope_acq_avg_mode_en_path0   :	True
#         scope_acq_avg_mode_en_path1   :	True
#         scope_acq_sequencer_select    :	0
#         scope_acq_trigger_level_path0 :	0
#         scope_acq_trigger_level_path1 :	0
#         scope_acq_trigger_mode_path0  :	sequencer
#         scope_acq_trigger_mode_path1  :	sequencer
#         cluster_module16_sequencer0:
#             parameter                       value
#         --------------------------------------------------------------------------------
#         channel_map_path0_out0_en        :	True
#         channel_map_path1_out1_en        :	True
#         cont_mode_en_awg_path0           :	False
#         cont_mode_en_awg_path1           :	False
#         cont_mode_waveform_idx_awg_path0 :	0
#         cont_mode_waveform_idx_awg_path1 :	0
#         demod_en_acq                     :	False
#         thresholded_acq_threshold        :	0
#         gain_awg_path0                   :	1
#         gain_awg_path1                   :	1
#         integration_length_acq           :	2000
#         marker_ovr_en                    :	True
#         marker_ovr_value                 :	15
#         mixer_corr_gain_ratio            :	1
#         mixer_corr_phase_offset_degree   :	-0
#         mod_en_awg                       :	False
#         nco_freq                         :	0 (Hz)
#         nco_phase_offs                   :	0 (Degrees)
#         offset_awg_path0                 :	0
#         offset_awg_path1                 :	0
#         thresholded_acq_rotation         :	0 (Degrees)
#         sequence                         :	/nfs/users/alvaro.orgaz/qibolab/src/qibola...
#         sync_en                          :	True
#         upsample_rate_awg_path0          :	0
#         upsample_rate_awg_path1          :	0

#     qcm:
#         platform.instruments['qcm_rf2'].device.print_readable_snapshot(update=True)
#         cluster_module12:
#             parameter        value
#         --------------------------------------------------------------------------------
#         out0_att          :	24 (dB)
#         out0_lo_en        :	True
#         out0_lo_freq      :	5325473000 (Hz)
#         out0_offset_path0 :	24 (mV)
#         out0_offset_path1 :	24 (mV)
#         out1_att          :	24 (dB)
#         out1_lo_en        :	True
#         out1_lo_freq      :	6212286000 (Hz)
#         out1_offset_path0 :	0 (mV)
#         out1_offset_path1 :	0 (mV)
#         present           :	True
#         cluster_module12_sequencer0:
#             parameter                       value
#         --------------------------------------------------------------------------------
#         channel_map_path0_out0_en        :	True
#         channel_map_path0_out2_en        :	False
#         channel_map_path1_out1_en        :	True
#         channel_map_path1_out3_en        :	False
#         cont_mode_en_awg_path0           :	False
#         cont_mode_en_awg_path1           :	False
#         cont_mode_waveform_idx_awg_path0 :	0
#         cont_mode_waveform_idx_awg_path1 :	0
#         gain_awg_path0                   :	0.33998
#         gain_awg_path1                   :	0.33998
#         marker_ovr_en                    :	True
#         marker_ovr_value                 :	15
#         mixer_corr_gain_ratio            :	1
#         mixer_corr_phase_offset_degree   :	-0
#         mod_en_awg                       :	False
#         nco_freq                         :	-2e+08 (Hz)
#         nco_phase_offs                   :	0 (Degrees)
#         offset_awg_path0                 :	0
#         offset_awg_path1                 :	0
#         sequence                         :	/nfs/users/alvaro.orgaz/qibolab/src/qibola...
#         sync_en                          :	True
#         upsample_rate_awg_path0          :	0
#         upsample_rate_awg_path1          :	0
#     """


# @pytest.mark.qpu
# @pytest.mark.parametrize("name", INSTRUMENTS_LIST)
# def test_instruments_process_pulse_sequence_upload_play(platform_name, name):
#     instrument = instruments[name]
#     settings = create_platform(platform_name).settings
#     instrument.setup(**settings["settings"], **instruments_settings[name])
#     relaxation_time = settings["settings"]["relaxation_time"]
#     instrument_pulses = {}
#     instrument_pulses[name] = PulseSequence()
#     if "QCM" in instrument.__class__.__name__:
#         for channel in instrument.channel_port_map:
#             instrument_pulses[name].add(Pulse(0, 200, 1, 10e6, np.pi / 2, "Gaussian(5)", str(channel)))
#         instrument.process_pulse_sequence(instrument_pulses[name], nshots=5, relaxation_time=relaxation_time)
#         instrument.upload()
#         instrument.play_sequence()
#     if "QRM" in instrument.__class__.__name__:
#         channel = instrument._port_channel_map["o1"]
#         instrument_pulses[name].add(
#             Pulse(0, 200, 1, 10e6, np.pi / 2, "Gaussian(5)", channel),
#             ReadoutPulse(200, 2000, 1, 10e6, np.pi / 2, "Rectangular()", channel),
#         )
#         instrument.device.sequencers[0].sync_en(
#             False
#         )  # TODO: Check why this is necessary here and not when playing a PS of only one readout pulse
#         instrument.process_pulse_sequence(instrument_pulses[name], nshots=5, relaxation_time=relaxation_time)
#         instrument.upload()
#         instrument.play_sequence()
#         acquisition_results = instrument.acquire()


# @pytest.mark.qpu
# @pytest.mark.parametrize("name", INSTRUMENTS_LIST)
# def test_instruments_qublox_start_stop_disconnect(name):
#     instrument = instruments[name]
#     instrument.start()
#     instrument.stop()
#     instrument.disconnect()
#     assert instrument.is_connected == False
