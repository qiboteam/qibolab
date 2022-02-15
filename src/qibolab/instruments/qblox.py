import json
import numpy as np
import matplotlib.pyplot as plt


class GenericPulsar:

    def __init__(self, sequencer=0, debugging=False):
        self.name = None
        self.sequencer = sequencer
        self.debugging = debugging

    def setup(self, gain, hardware_avg, initial_delay, repetition_duration):
        if self.sequencer == 1:
            self.device.sequencer1_gain_awg_path0(gain)
            self.device.sequencer1_gain_awg_path1(gain)
        else:
            self.sequencer0_gain_awg_path0(gain)
            self.sequencer0_gain_awg_path1(gain)
        self.hardware_avg = hardware_avg
        self.initial_delay = initial_delay
        self.repetition_duration = repetition_duration
        self.acquisitions = {"single": {"num_bins": 1, "index":0}}
        self.weights = {}

    def _translate_single_pulse(self, pulse):
        # Use the envelope to modulate a sinusoldal signal of frequency freq_if
        envelopes   = np.stack(pulse.envelopes(), axis=0)
        time        = np.arange(pulse.length) * 1e-9
        cosalpha    = np.cos(2 * np.pi * pulse.frequency * time + pulse.phase)
        sinalpha    = np.sin(2 * np.pi * pulse.frequency * time + pulse.phase)
        mod_matrix  = np.array([[ cosalpha, sinalpha],
                                [-sinalpha, cosalpha]])
        mod_signals = np.einsum("abt,bt->ta", mod_matrix, envelopes)

        # add offsets to compensate mixer leakage
        waveform = {
                "modI": {"data": mod_signals[:, 0] + pulse.offset_i, "index": 0},
                "modQ": {"data": mod_signals[:, 1] + pulse.offset_q, "index": 1}
            }

        if self.debugging:
            # Plot the result
            fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
            ax.plot(waveform["modI"]["data"],'-',color='C0')
            ax.plot(waveform["modQ"]["data"],'-',color='C1')
            ax.title.set_text('pulse')

        return waveform

    def translate(self, sequence):
        pulses = list(sequence)
        if sequence.readout_pulse is not None:
            pulses.append(sequence.readout_pulse)
        if not pulses:
            raise NotImplementedError("Cannot translate empty sequence.")

        name = self.name
        waveform = self._translate_single_pulse(pulses[0])
        waveforms = {
            f"modI_{name}": {"data": [], "index": 0},
            f"modQ_{name}": {"data": [], "index": 1}
        }
        waveforms[f"modI_{name}"]["data"] = waveform.get("modI").get("data")
        waveforms[f"modQ_{name}"]["data"] = waveform.get("modQ").get("data")

        for pulse in sequence[1:]:
            waveform = self._translate_single_pulse(pulse)
            waveforms[f"modI_{name}"]["data"] = np.concatenate((waveforms[f"modI_{name}"]["data"], np.zeros(4), waveform["modI"]["data"]))
            waveforms[f"modQ_{name}"]["data"] = np.concatenate((waveforms[f"modQ_{name}"]["data"], np.zeros(4), waveform["modQ"]["data"]))

        if self.debugging:
            # Plot the result
            fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
            ax.plot(waveforms[f"modI_{name}"]["data"], '-', color='C0')
            ax.plot(waveforms[f"modQ_{name}"]["data"], '-', color='C1')
            ax.title.set_text('Combined Pulses')

        program = self.generate_program(sequence.start, sequence.readout_pulse)
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

    def upload(self, waveforms, program, data_folder):
        import os
        # Upload waveforms and program
        # Reformat waveforms to lists
        for name, waveform in waveforms.items():
            if isinstance(waveform["data"], np.ndarray):
                waveforms[name]["data"] = waveforms[name]["data"].tolist()  # JSON only supports lists

        #Add sequence program and waveforms to single dictionary and write to JSON file
        filename = f"{data_folder}/qrm_sequence.json"
        program_dict = {
            "waveforms": waveforms,
            "weights": self.weights,
            "acquisitions": self.acquisitions,
            "program": program
            }
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(program_dict, file, indent=4)

        # Upload json file to the device
        if self.sequencer == 1:
            self.sequencer1_waveforms_and_program(os.path.join(os.getcwd(), filename))
        else:
            self.sequencer0_waveforms_and_program(os.path.join(os.getcwd(), filename))

    def play_sequence(self):
        # arm sequencer and start playing sequence
        self.device.arm_sequencer()
        self.device.start_sequencer()
        if self.debugging:
            print(self.device.get_sequencer_state(self.sequencer))


class PulsarQRM(GenericPulsar):
    """Class for interfacing with Pulsar QRM."""

    def __init__(self, label, ip,
                 ref_clock="external", sequencer=0, sync_en=True,
                 hardware_avg_en=True, acq_trigger_mode="sequencer",
                 debugging=False):
        from pulsar_qrm.pulsar_qrm import pulsar_qrm
        super().__init__(sequencer, debugging)
        # Instantiate base object from qblox library and connect to it
        self.device = pulsar_qrm(label, ip)
        self.name = "qrm"

        # Reset and configure
        self.device.reset()
        self.device.reference_source(ref_clock)
        self.device.scope_acq_sequencer_select(sequencer)
        self.device.scope_acq_avg_mode_en_path0(hardware_avg_en)
        self.device.scope_acq_avg_mode_en_path1(hardware_avg_en)
        self.device.scope_acq_trigger_mode_path0(acq_trigger_mode)
        self.device.scope_acq_trigger_mode_path1(acq_trigger_mode)
        # sync sequencer
        if self.sequencer == 1:
            self.device.sequencer1_sync_en(sync_en)
        else:
            self.device.sequencer0_sync_en(sync_en)

    def setup(self, gain, hardware_avg, initial_delay, repetition_duration,
              start_sample, integration_length, sampling_rate, mode):
        super().setup(gain, hardware_avg, initial_delay, repetition_duration)
        self.start_sample = start_sample
        self.integration_length = integration_length
        self.sampling_rate = sampling_rate
        self.mode = mode

    def play_sequence_and_acquire(self, ro_pulse):
        #arm sequencer and start playing sequence
        super().play_sequence()
        #start acquisition of data
        #Wait for the sequencer to stop with a timeout period of one minute.
        self.device.get_sequencer_state(0, 1)
        #Wait for the acquisition to finish with a timeout period of one second.
        self.device.get_acquisition_state(self.sequencer, 1)
        #Move acquisition data from temporary memory to acquisition list.
        self.device.store_scope_acquisition(self.sequencer, "single")
        #Get acquisition list from instrument.
        single_acq = self.device.get_acquisitions(self.sequencer)
        if self.debugging:
            self._plot_acquisitions(single_acq)
            with open(".data/results.json", 'w', encoding='utf-8') as file:
                json.dump(single_acq, file, indent=4)
        i, q = self._demodulate_and_integrate(single_acq, ro_pulse)
        acquisition_results = np.sqrt(i**2 + q**2), np.arctan2(q, i), i, q
        return acquisition_results

    @staticmethod
    def _plot_acquisitions(single_acq):
        #Plot acquired signal on both inputs I and Q
        fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
        ax.plot(single_acq["single"]["acquisition"]["scope"]["path0"]["data"][120:6500])
        ax.plot(single_acq["single"]["acquisition"]["scope"]["path1"]["data"][120:6500])
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Relative amplitude')
        plt.show()

    def _demodulate_and_integrate(self, single_acq, ro_pulse):
        #DOWN Conversion
        norm_factor = 1. / (self.integration_length)
        n0 = self.start_sample
        n1 = self.start_sample + self.integration_length
        input_vec_I = np.array(single_acq["single"]["acquisition"]["scope"]["path0"]["data"][n0: n1])
        input_vec_Q = np.array(single_acq["single"]["acquisition"]["scope"]["path1"]["data"][n0: n1])
        input_vec_I -= np.mean(input_vec_I)
        input_vec_Q -= np.mean(input_vec_Q)

        if self.mode == 'ssb':
            modulated_i = input_vec_I
            modulated_q = input_vec_Q
            time = np.arange(modulated_i.shape[0])*1e-9
            cosalpha = np.cos(2 * np.pi * ro_pulse.frequency * time)
            sinalpha = np.sin(2 * np.pi * ro_pulse.frequency * time)
            demod_matrix = 2 * np.array([[cosalpha, -sinalpha], [sinalpha, cosalpha]])
            result = []
            for it, t, ii, qq in zip(np.arange(modulated_i.shape[0]), time,modulated_i, modulated_q):
                result.append(demod_matrix[:,:,it] @ np.array([ii, qq]))
            demodulated_signal = np.array(result)
            integrated_signal = norm_factor*np.sum(demodulated_signal,axis=0)
            if self.debugging:
                print(integrated_signal,demodulated_signal[:,0].max()-demodulated_signal[:,0].min(),demodulated_signal[:,1].max()-demodulated_signal[:,1].min())
        elif self.mode == 'optimal':
            raise NotImplementedError('Optimal Demodulation Mode not coded yet.')
        else:
            raise NotImplementedError('Demodulation mode not understood.')
        return integrated_signal


class PulsarQCM(GenericPulsar):

    def __init__(self, label, ip,
                 ref_clock="external", sequencer=0, sync_en=True,
                 debugging=False):
        from pulsar_qcm.pulsar_qcm import pulsar_qcm
        super().__init__(sequencer, debugging)
        # Instantiate base object from qblox library and connect to it
        self.device = pulsar_qcm(label, ip)
        self.name = "qcm"
        # Reset and configure
        self.device.reset()
        self.device.reference_source(ref_clock)
        if self.sequencer == 1:
            self.device.sequencer1_sync_en(sync_en)
        else:
            self.device.sequencer0_sync_en(sync_en)
