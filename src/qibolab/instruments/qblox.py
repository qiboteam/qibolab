import json
import numpy as np
from abc import ABC, abstractmethod


class GenericPulsar(ABC):

    def __init__(self):
        # To be defined in each instrument
        self.name = None
        self.device = None
        self.sequencer = None
        self.ref_clock = None
        self.sync_en = None
        # To be defined during setup
        self.hardware_avg = None
        self.initial_delay = None
        self.repetition_duration = None
        # hardcoded values used in ``generate_program``
        self.delay_before_readout = 4 # same value is used for all readout pulses (?)
        self.wait_loop_step = 1000
        self.duration_base = 16380 # maximum length of a waveform in number of samples (defined by the device memory).
        # hardcoded values used in ``upload``

    def setup(self, gain, hardware_avg, initial_delay, repetition_duration):
        if self.sequencer == 1:
            self.device.sequencer1_gain_awg_path0(gain)
            self.device.sequencer1_gain_awg_path1(gain)
        else:
            self.device.sequencer0_gain_awg_path0(gain)
            self.device.sequencer0_gain_awg_path1(gain)
        self.hardware_avg = hardware_avg
        self.initial_delay = initial_delay
        self.repetition_duration = repetition_duration

    def _translate_single_pulse(self, pulse):
        # Use the envelope to modulate a sinusoldal signal of frequency freq_if
        envelope_i = pulse.compile()
        # TODO: if ``envelope_q`` is not always 0 we need to find how to
        # calculate it
        envelope_q = np.zeros(int(self.length))
        time = np.arange(pulse.length) * 1e-9
        # FIXME: There should be a simpler way to construct this array
        cosalpha = np.cos(2 * np.pi * pulse.frequency * time)
        sinalpha = np.sin(2 * np.pi * pulse.frequency * time)
        mod_matrix = np.array([[cosalpha,sinalpha], [-sinalpha,cosalpha]])
        result = []
        for it, t, ii, qq in zip(np.arange(pulse.length), time, envelope_i, envelope_q):
            result.append(mod_matrix[:, :, it] @ np.array([ii, qq]))
        mod_signals = np.array(result)

        # add offsets to compensate mixer leakage
        waveform = {
            "modI": {"data": mod_signals[:, 0] + pulse.offset_i, "index": 0},
            "modQ": {"data": mod_signals[:, 1] + pulse.offset_q, "index": 1}
        }
        return waveform

    def generate_waveforms(self, pulses):
        if not pulses:
            raise_error(NotImplementedError, "Cannot translate empty pulse sequence.")
        name = self.name
        waveform = self._translate_single_pulse(pulses[0])
        waveforms = {
            f"modI_{name}": {"data": [], "index": 0},
            f"modQ_{name}": {"data": [], "index": 1}
        }
        waveforms[f"modI_{name}"]["data"] = waveform.get("modI").get("data")
        waveforms[f"modQ_{name}"]["data"] = waveform.get("modQ").get("data")
        for pulse in pulses[1:]:
            waveform = self._translate_single_pulse(pulse)
            waveforms[f"modI_{name}"]["data"] = np.concatenate((waveforms[f"modI_{name}"]["data"], np.zeros(4), waveform["modI"]["data"]))
            waveforms[f"modQ_{name}"]["data"] = np.concatenate((waveforms[f"modQ_{name}"]["data"], np.zeros(4), waveform["modQ"]["data"]))
        return waveforms

    def generate_program(self, initial_delay, acquire_instruction, wait_time):
        """Generates the program to be uploaded to instruments."""
        extra_duration = self.repetition_duration - self.duration_base
        extra_wait = extra_duration % wait_loop_step
        num_wait_loops = (extra_duration - extra_wait) // self. wait_loop_step

        # This calculation was moved to `PulsarQCM` and `PulsarQRM`
        #if ro_pulse is not None:
        #    acquire_instruction = "acquire   0,0,4"
        #    wait_time = self.duration_base - initial_delay - delay_before_readout - 4
        #else:
        #    acquire_instruction = ""
        #    wait_time = self.duration_base - initial_delay - delay_before_readout

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
            play      0,1,{self.delay_before_readout}
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
        return program

    @abstractmethod
    def translate(self, sequence):
        raise_error(NotImplementedError)

    def upload(self, waveforms, program, data_folder):
        import os
        # Upload waveforms and program
        # Reformat waveforms to lists
        for name, waveform in waveforms.items():
            if isinstance(waveform["data"], np.ndarray):
                waveforms[name]["data"] = waveforms[name]["data"].tolist()  # JSON only supports lists

        # Add sequence program and waveforms to single dictionary and write to JSON file
        filename = f"{data_folder}/qrm_sequence.json"
        program_dict = {
            "waveforms": waveforms,
            "weights": self.weights,
            "acquisitions": self.acquisitions,
            "program": program
            }
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(program_dict, file, indent=4)

        # Upload json file to the device
        if self.sequencer == 1:
            self.device.sequencer1_waveforms_and_program(os.path.join(os.getcwd(), filename))
        else:
            self.device.sequencer0_waveforms_and_program(os.path.join(os.getcwd(), filename))

    def play_sequence(self):
        # arm sequencer and start playing sequence
        self.device.arm_sequencer()
        self.device.start_sequencer()

    def stop(self):
        self.device.stop_sequencer()

    def close(self):
        self.device.close()

    def __del__(self):
        self.device.close()


class PulsarQRM(GenericPulsar):
    """Class for interfacing with Pulsar QRM."""

    def __init__(self, label, ip, ref_clock="external", sequencer=0, sync_en=True,
                 hardware_avg_en=True, acq_trigger_mode="sequencer"):
        from pulsar_qrm.pulsar_qrm import pulsar_qrm # pylint: disable=E0401
        super().__init__()
        # Instantiate base object from qblox library and connect to it
        self.name = "qrm"
        self.device = pulsar_qrm(label, ip)
        self.sequencer = sequencer
        self.hardware_avg_en = hardware_avg_en

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

    def translate(self, sequence):
        # Allocate only readout pulses to PulsarQRM
        waveforms = self.generate_waveforms(sequence.qrm_pulses)

        # Generate program without acquire instruction
        initial_delay = sequence.qrm_pulses[0].start
        # Acquire waveforms over remaining duration of acquisition of input vector of length = 16380 with integration weights 0,0
        acquire_instruction = "acquire   0,0,4"
        wait_time = self.duration_base - initial_delay - self.delay_before_readout - 4 # FIXME: Not sure why this hardcoded 4 is needed
        program = self.generate_program(initial_delay, acquire_instruction, wait_time)

        return waveforms, program

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
        i, q = self._demodulate_and_integrate(single_acq, ro_pulse)
        acquisition_results = np.sqrt(i**2 + q**2), np.arctan2(q, i), i, q
        return acquisition_results

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

        elif self.mode == 'optimal':
            raise_error(NotImplementedError, "Optimal Demodulation Mode not coded yet.")
        else:
            raise_error(NotImplementedError, "Demodulation mode not understood.")
        return integrated_signal


class PulsarQCM(GenericPulsar):

    def __init__(self, label, ip, sequencer=0, ref_clock="external", sync_en=True):
        from pulsar_qcm.pulsar_qcm import pulsar_qcm # pylint: disable=E0401
        super().__init__()
        # Instantiate base object from qblox library and connect to it
        self.name = "qcm"
        self.device = pulsar_qcm(label, ip)
        self.sequencer = sequencer
        # Reset and configure
        self.device.reset()
        self.device.reference_source(ref_clock)
        if self.sequencer == 1:
            self.device.sequencer1_sync_en(sync_en)
        else:
            self.device.sequencer0_sync_en(sync_en)

    def translate(self, sequence):
        # Allocate only qubit pulses to PulsarQRM
        waveforms = self.generate_waveforms(sequence.qcm_pulses)

        # Generate program without acquire instruction
        initial_delay = sequence.qcm_pulses[0].start
        acquire_instruction = ""
        wait_time = self.duration_base - initial_delay - self.delay_before_readout
        program = self.generate_program(initial_delay, acquire_instruction, wait_time)

        return waveforms, program
