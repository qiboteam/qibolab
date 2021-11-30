from pulsar_qcm.pulsar_qcm import pulsar_qcm
from pulsar_qrm.pulsar_qrm import pulsar_qrm


class PulsarQRM(pulsar_qrm):
    """Class for interfacing with Pulsar QRM."""

    def __init__(self, label, ip,
                 ref_clock="external", sequencer=0, sync_en=True,
                 hardware_avg_en=True, acq_trigger_mode="sequencer",
                 debugging=False):
        # Instantiate base object from qblox library and connect to it
        super().__init__(label, ip)

        # Reset and configure
        self.reset()
        self.reference_source(ref_clock)
        self.scope_acq_sequencer_select(sequencer)
        self.scope_acq_avg_mode_en_path0(hardware_avg_en)
        self.scope_acq_avg_mode_en_path1(hardware_avg_en)
        self.scope_acq_trigger_mode_path0(acq_trigger_mode)
        self.scope_acq_trigger_mode_path1(acq_trigger_mode)

        self.sequencer = sequencer
        if self.sequencer == 1:
            self.sequencer1_sync_en(sync_en)
        else:
            self.sequencer0_sync_en(sync_en)

        self.debugging = debugging

    def setup(self, gain, hardware_avg, initial_delay, repetition_duration,
              start_sample, integration_length, sampling_rate, mode):
        if self.sequencer == 1:
            self.sequencer1_gain_awg_path0(gain)
            self.sequencer1_gain_awg_path1(gain)
        else:
            self.sequencer0_gain_awg_path0(gain)
            self.sequencer0_gain_awg_path1(gain)
        self.hardware_avg = hardware_avg
        self.initial_delay = initial_delay
        self.repetition_duration = repetition_duration

    def translate(self, pulses):
        waveform = pulses[0].waveform()
        waveforms = {
            "modI_qrm": {"data": [], "index": 0},
            "modQ_qrm": {"data": [], "index": 1}
        }
        waveforms["modI_qrm"]["data"] = waveform.get("modI").get("data")
        waveforms["modQ_qrm"]["data"] = waveform.get("modQ").get("data")

        for pulse in pulses[1:]:
            waveform = pulse.waveform()
            waveforms["modI_qrm"]["data"] = np.concatenate((waveforms["modI_qrm"]["data"], np.zeros(4), waveform["modI"]["data"]))
            waveforms["modQ_qrm"]["data"] = np.concatenate((waveforms["modQ_qrm"]["data"], np.zeros(4), waveform["modQ"]["data"]))

        if self.debugging:
            # Plot the result
            fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
            ax.plot(combined_waveforms["modI_qrm"]["data"],'-',color='C0')
            ax.plot(combined_waveforms["modQ_qrm"]["data"],'-',color='C1')
            ax.title.set_text('Combined Pulses')

        ro_pulse = None # TODO: We can use PulseSequence.readout_pulse
        for pulse in pulses:
            if pulse.name == "ro_pulse": # isinstance(pulse, ReadoutPulse)
                ro_pulse = pulse
        program = self.generate_program(pulses[0].start, ro_pulse)
        return waveforms, program

    @staticmethod
    def calculate_repetition_rate(self, repetition_duration,
                                  wait_loop_step, duration_base):
        extra_duration = repetition_duration-duration_base
        extra_wait = extra_duration % wait_loop_step
        num_wait_loops = (extra_duration - extra_wait) // wait_loop_step
        return num_wait_loops, extra_wait

    def generate_program(self, initial_delay, ro_pulse=None):
        # Prepare sequence program
        wait_loop_step=1000
        duration_base=16380 # this is the maximum length of a waveform in number of samples (defined by the device memory)

        num_wait_loops, extra_wait = calculate_repetition_rate(self.repetition_duration, wait_loop_step, duration_base)
        if ro_pulse is not None:
            delay_before_readout = ro_pulse.delay_before_readout
            acquire_instruction = "acquire   0,0,4      # Acquire waveforms over remaining duration of acquisition of input vector of length = 16380 with integration weights 0,0"
            wait_time = duration_base - initial_delay - delay_before_readout - 4 # pulses['ro_pulse']['start']
        else:
            delay_before_readout = 4
            acquire_instruction = ""
            wait_time = duration_base - initial_delay - delay_before_readout

        if initial_delay != 0:
            initial_wait_instruction = f"wait      {initial_delay}"
        else:
            initial_wait_instruction = ""
        program = f"""
            move    {self.hardware_avg},R0
            nop
            wait_sync 4          # Synchronize sequencers over multiple instruments
        loop:
            {initial_wait_instruction}
            play      0,1,{delay_before_readout}
            {acquire_instruction}
            wait      {wait_time}
            move      {num_wait_loops},R1
            nop
            repeatloop:
                wait      {wait_loop_step}
                loop      R1,@repeatloop
            wait      {extra_wait}
            loop    R0,@loop
            stop
        """
        if self.debugging:
            print(program)

        return program

    def get_acquisitions(self, acquisitions={"single": {"num_bins": 1, "index":0}}):
        return acquisitions

    def get_weights(self, weights = {}):
        return weights

    def upload(self, waveforms, program, acquisitions, weights, data_folder):
        import os
        # Upload waveforms and program
        # Reformat waveforms to lists
        for name, waveform in waveforms.items():
            if isinstance(waveform["data"], np.ndarray):
                waveforms[name]["data"] = waveforms[name]["data"].tolist()  # JSON only supports lists

        #Add sequence program and waveforms to single dictionary and write to JSON file
        filename = f"{data_folder}/qrm_sequence.json"
        wave_and_prog_dict = {
            "waveforms": waveforms,
            "weights": weights,
            "acquisitions": acquisitions,
            "program": program
            }
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(wave_and_prog_dict, file, indent=4)

        # Upload json file to the device
        if self.sequencer == 1:
            self.sequencer1_waveforms_and_program(os.path.join(os.getcwd(), filename))
        else:
            self.sequencer0_waveforms_and_program(os.path.join(os.getcwd(), filename))


class PulsarQCM(pulsar_qcm):

    def __init__(self, label, ip,
                 ref_clock="external", sequencer=0, sync_en=True):
        # Instantiate base object from qblox library and connect to it
        super().__init__(label, ip)

        # Reset and configure
        self.reset()
        self.reference_source(ref_clock)

        self.sequencer = sequencer
        if self.sequencer == 1:
            self.sequencer1_sync_en(sync_en)
        else:
            self.sequencer0_sync_en(sync_en)

    def setup(self, gain=0.5):
        if self.sequencer == 1:
            self.sequencer1_gain_awg_path0(gain)
            self.sequencer1_gain_awg_path1(gain)
        else:
            self.sequencer0_gain_awg_path0(gain)
            self.sequencer0_gain_awg_path1(gain)
