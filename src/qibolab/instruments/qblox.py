"""
Class for interfacing with Qblox Qubit Control and Readout Modules (Pulsar QCM & Pulsar QRM)
"""
import os
import scipy.signal
#import math
import json
import numpy as np
import matplotlib.pyplot as plt
#import lmfit
#from scipy.signal import waveforms
import xarray as xr

# qblox-instruments libraries. this code was tested against qblox-instruments 0.4.0
from pulsar_qcm.pulsar_qcm import pulsar_qcm
from pulsar_qrm.pulsar_qrm import pulsar_qrm

debugging = False

def generate_waveforms(pulse):
    """
    This function generates the I & Q waveforms to be sent to the sequencers based on the key parameters of the pulse (length, amplitude, shape, etc.)
    """
    freq_if = pulse["freq_if"]      # Pulse Intermediate Frequency in Hz [10e6 to 300e6]
    amplitude = pulse["amplitude"]  # Pulse digital amplitude (unitless) [0 to 1]
    length = pulse["length"]        # pulse duration in ns
    offset_i = pulse["offset_i"]    # Pulse I offset (unitless). (amplitude + offset) should be between [0 and 1]
    offset_q = pulse["offset_q"]    # Pulse Q offset (unitless). (amplitude + offset) should be between [0 and 1]
    shape = pulse["shape"]          # Pulse shape ['Block', 'Gaussian']

    # Generate pulse envelope
    if shape == 'Block':
        envelope_i = amplitude*np.ones(int(length))
        envelope_q = amplitude*np.zeros(int(length))
    elif shape == 'Gaussian':
        std = length/5
        envelope_i = amplitude*scipy.signal.gaussian(length, std=std)
        envelope_q = amplitude*np.zeros(int(length))

    # Use the envelope to modulate a sinusoldal signal of frequency freq_if
    time = np.arange(length)*1e-9
    cosalpha = np.cos(2*np.pi*freq_if*time)
    sinalpha = np.sin(2*np.pi*freq_if*time)
    mod_matrix = np.array([[cosalpha,sinalpha],[-sinalpha,cosalpha]])
    result = []
    for it,t,ii,qq in zip(np.arange(length),time,envelope_i,envelope_q):
        result.append(mod_matrix[:,:,it]@np.array([ii,qq]))
    mod_signals = np.array(result)

    waveforms = {
            "modI": {"data": [], "index": 0},
            "modQ": {"data": [], "index": 1}
        }
    # add offsets to compensate mixer leakage
    waveforms["modI"]["data"] = mod_signals[:,0]+offset_i
    waveforms["modQ"]["data"] = mod_signals[:,1]+offset_q

    if debugging:
        # Plot the result
        fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
        ax.plot(waveforms["modI"]["data"],'-',color='C0')
        ax.plot(waveforms["modQ"]["data"],'-',color='C1')
        ax.title.set_text('pulse')
    return waveforms

def calculate_repetition_rate(repetition_duration,
                            wait_loop_step,
                            duration_base):
    extra_duration = repetition_duration-duration_base
    extra_wait = extra_duration%wait_loop_step
    num_wait_loops = (extra_duration-extra_wait)//wait_loop_step
    return num_wait_loops,extra_wait

def generate_program(program_parameters):
    # Prepare sequence program
    wait_loop_step=1000
    duration_base=16380 # this is the maximum length of a waveform in number of samples (defined by the device memory)
    hardware_avg = program_parameters["hardware_avg"]
    initial_delay = program_parameters["initial_delay"]
    repetition_duration= program_parameters["repetition_duration"]
    pulses = program_parameters['pulses']
    
    num_wait_loops,extra_wait = calculate_repetition_rate(repetition_duration, wait_loop_step, duration_base)
    if 'ro_pulse' in pulses:
        acquire_instruction = "acquire   0,0,4      # Acquire waveforms over remaining duration of acquisition of input vector of length = 16380 with integration weights 0,0"
        pause = pulses['ro_pulse']['start']
    else:
        acquire_instruction = ""
        pause = 4

    if initial_delay != 0: 
        initial_wait_instruction = f"wait      {initial_delay}"
    else:
        initial_wait_instruction = ""
    program = f"""
        move    {hardware_avg},R0
        nop
        wait_sync 4          # Synchronize sequencers over multiple instruments

    loop:
        {initial_wait_instruction}           
        play      0,1,{pause}      
        {acquire_instruction}
        wait      {duration_base-initial_delay-pause}
        move      {num_wait_loops},R1     
        nop
        repeatloop:
            wait      {wait_loop_step}
            loop      R1,@repeatloop
        wait      {extra_wait}
        loop    R0,@loop

        stop
    """
    if debugging:
        print(program)

    return program


class Pulsar_QRM():
    """
    Class for interfacing with Pulsar QRM. It implements Quantify Gettable Interface to allow for real time plotting
    """
    def __init__(self, label, ip, settings = {}):
        """
        This method connects to the Pulsar QRM and configures it with those settings 
        that are not expected to change.
        All parameters within settings are optional, their default values are:
            QRM_init_settings = {
                    'ref_clock': 'external',                        # Clock source ['external', 'internal']
                    'sequencer': 0,
                    'sync_en': True,
                    'hardware_avg_en': True,                        # If enabled, the device repeats the pulse multiple times (hardware_avg)
                    'acq_trigger_mode': 'sequencer',
            }
        """
        settings.setdefault('ref_clock', 'external')
        settings.setdefault('sequencer', 0)
        settings.setdefault('sync_en', True)
        settings.setdefault('hardware_avg_en', True)
        settings.setdefault('acq_trigger_mode', 'sequencer')

        # Instantiate base object from qblox library and connect to it
        qrm = pulsar_qrm(label, ip)

        # Reset and configure
        qrm.reset()
        qrm.reference_source(settings['ref_clock'])

        qrm.scope_acq_sequencer_select(settings['sequencer'])
        qrm.scope_acq_avg_mode_en_path0(settings['hardware_avg_en'])
        qrm.scope_acq_avg_mode_en_path1(settings['hardware_avg_en'])

        qrm.scope_acq_trigger_mode_path0(settings['acq_trigger_mode'])
        qrm.scope_acq_trigger_mode_path1(settings['acq_trigger_mode'])

        if settings['sequencer'] == 1:
            qrm.sequencer1_sync_en(settings['sync_en'])
        else:
            qrm.sequencer0_sync_en(settings['sync_en'])
            
        self._qrm = qrm
        self._settings = settings

    def setup(self, settings = {}):
        """
        This method connects to the device and configures it with the settings provided
            All parameters are optional, their default values are:
            QRM_settings = {
                    'gain': 0.5,                                    # Analog amplification gain [0 - 1]
                    'hardware_avg': 1024,

                    'ro_pulse': {	"freq_if": 20e6,                # Pulse Intermediate Frequency in Hz [10e6 to 300e6]
                                    "amplitude": 0.5,               # Pulse digital amplitude (unitless) [0 to 1]
                                    "length": 6000,                 # pulse duration in ns
                                    "offset_i": 0,                  # Pulse I offset (unitless). (amplitude + offset) should be between [0 and 1]
                                    "offset_q": 0,                  # Pulse Q offset (unitless). (amplitude + offset) should be between [0 and 1]
                                    "shape": "Block",               # Pulse shape ['Block', 'Gaussian']
                                    "delay_before": 344,            # Delay before the bulse in ns
                                    "repetition_duration": 200000,  # Total time between pulses = delay before + pulse length + delay after
                                                },

                    'start_sample': 130,                            # Number of samples to skip before integration starts
                    'integration_length': 5000,                     # Number of samples to integrate over (start_sample + integration_length) < pulse length
                    'sampling_rate': 1e9,                           # device sampling rate in samples per second
                    'mode': 'ssb',                                  # demodulation mode ['ssb', ...]
            }
        """

        #settings.setdefault('gain', 0.5)
        #settings.setdefault('hardware_avg', 1024)

        #settings.setdefault('ro_pulse', {"freq_if": 20e6,
        #                                "amplitude": 0.5,
        #                                "length": 6000,
        #                                "offset_i": 0,
        #                                "offset_q": 0,
        #                                "shape": "Block",
        #                                "delay_before": 344,
        #                                "repetition_duration": 200000,
        #                              })

        #settings.setdefault('start_sample', 130)
        #settings.setdefault('integration_length', 5000)
        #settings.setdefault('sampling_rate', 1e9)
        #settings.setdefault('mode', 'ssb')        

        self._settings.update(settings)

        qrm = self._qrm
        sequencer = self._settings['sequencer']

        if sequencer == 1:
            qrm.sequencer1_gain_awg_path0(settings['gain'])
            qrm.sequencer1_gain_awg_path1(settings['gain'])
        else:
            qrm.sequencer0_gain_awg_path0(settings['gain'])
            qrm.sequencer0_gain_awg_path1(settings['gain'])

    def set_waveforms(self, waveforms):
        self._waveforms = waveforms
    def set_waveforms_from_pulses_definition(self, pulses_definition: dict):
        pulses_list = list(pulses_definition.values())
        pulse_waveforms = generate_waveforms(pulses_list.pop(0))
        combined_waveforms = {
            "modI_qrm": {"data": [], "index": 0},
            "modQ_qrm": {"data": [], "index": 1}
        }
        combined_waveforms["modI_qrm"]["data"] = pulse_waveforms["modI"]["data"]
        combined_waveforms["modQ_qrm"]["data"] = pulse_waveforms["modQ"]["data"]

        for pulse in pulses_list:
            pulse_waveforms = generate_waveforms(pulse)
            combined_waveforms["modI_qrm"]["data"] = np.concatenate((combined_waveforms["modI_qrm"]["data"],np.zeros(4), pulse_waveforms["modI"]["data"]))
            combined_waveforms["modQ_qrm"]["data"] = np.concatenate((combined_waveforms["modQ_qrm"]["data"],np.zeros(4), pulse_waveforms["modQ"]["data"]))

        self._waveforms = combined_waveforms
        if debugging:
            # Plot the result
            fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
            ax.plot(combined_waveforms["modI_qrm"]["data"],'-',color='C0')
            ax.plot(combined_waveforms["modQ_qrm"]["data"],'-',color='C1')
            ax.title.set_text('Combined Pulses')

    def set_program(self, program):
        self._program = program
    def set_program_from_parameters(self, parameters):
        self._program = generate_program(parameters)
    
    def set_acquisitions(self, acquisitions = {"single":   {"num_bins": 1, "index":0}}):
        self._acquisitions = acquisitions
    def set_weights(self, weights = {}):
        self._weights = weights
    
    def upload_sequence(self):    
        waveforms = self._waveforms
        program = self._program
        acquisitions = self._acquisitions
        weights = self._weights
        qrm = self._qrm
        sequencer = self._settings['sequencer']

        # Upload waveforms and program
        # Reformat waveforms to lists
        for name in waveforms:
            if str(type(waveforms[name]["data"]).__name__) == "ndarray":
                waveforms[name]["data"] = waveforms[name]["data"].tolist()  # JSON only supports lists

        #Add sequence program and waveforms to single dictionary and write to JSON file
        wave_and_prog_dict = {"waveforms": waveforms, "weights": weights, "acquisitions": acquisitions, "program": program}
        with open(".data/qrm_sequence.json", 'w', encoding='utf-8') as file:
            json.dump(wave_and_prog_dict, file, indent=4)
            file.close()

        #Upload json file to the device
        if sequencer == 1:
            qrm.sequencer1_waveforms_and_program(os.path.join(os.getcwd(), ".data/qrm_sequence.json"))
        else:
            qrm.sequencer0_waveforms_and_program(os.path.join(os.getcwd(), ".data/qrm_sequence.json"))

        self._wave_and_prog_dict = wave_and_prog_dict

    def play_sequence_and_acquire(self):
        qrm = self._qrm
        sequencer = self._settings['sequencer']

        #arm sequencer and start playing sequence
        qrm.arm_sequencer()
        qrm.start_sequencer()
        if debugging:
            print(qrm.get_sequencer_state(sequencer))
        #start acquisition of data
        #Wait for the sequencer to stop with a timeout period of one minute.
        qrm.get_sequencer_state(0, 1)
        #Wait for the acquisition to finish with a timeout period of one second.
        qrm.get_acquisition_state(sequencer, 1)
        #Move acquisition data from temporary memory to acquisition list.
        qrm.store_scope_acquisition(sequencer, "single")
        #Get acquisition list from instrument.
        self._single_acq = qrm.get_acquisitions(sequencer)
        if debugging:
            self._plot_acquisitions()
        i,q = self._demodulate_and_integrate()
        acquisition_results = np.sqrt(i**2+q**2),np.arctan2(q,i),i,q
        self._acquisition_results = acquisition_results
        return acquisition_results

    def _plot_acquisitions(self):
        #Plot acquired signal on both inputs I and Q
        fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
        ax.plot(self._single_acq["single"]["acquisition"]["scope"]["path0"]["data"][120:6500])
        ax.plot(self._single_acq["single"]["acquisition"]["scope"]["path1"]["data"][120:6500])
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Relative amplitude')
        plt.show()

    def _demodulate_and_integrate(self):
        settings = self._settings
        freq_if = settings['pulses']['ro_pulse']['freq_if']
        start_sample =   settings['start_sample']
        integration_length = settings['integration_length']
        sampling_rate = settings['sampling_rate']
        mode = settings['mode']

        #DOWN Conversion
        norm_factor = 1./(integration_length)
        input_vec_I = np.array(self._single_acq["single"]["acquisition"]["scope"]["path0"]["data"][start_sample:start_sample+integration_length])
        input_vec_Q = np.array(self._single_acq["single"]["acquisition"]["scope"]["path1"]["data"][start_sample:start_sample+integration_length])
        input_vec_I -= np.mean(input_vec_I)
        input_vec_Q -= np.mean(input_vec_Q)

        if mode == 'ssb':
            modulated_i = input_vec_I
            modulated_q = input_vec_Q
            time = np.arange(modulated_i.shape[0])*1e-9
            cosalpha = np.cos(2*np.pi*freq_if*time)
            sinalpha = np.sin(2*np.pi*freq_if*time)
            demod_matrix = 2*np.array([[cosalpha,-sinalpha],[sinalpha,cosalpha]])
            result = []
            for it,t,ii,qq in zip(np.arange(modulated_i.shape[0]),time,modulated_i,modulated_q):
                result.append(demod_matrix[:,:,it]@np.array([ii,qq]))
            demodulated_signal = np.array(result)
            integrated_signal = norm_factor*np.sum(demodulated_signal,axis=0)
            if debugging:
                print(integrated_signal,demodulated_signal[:,0].max()-demodulated_signal[:,0].min(),demodulated_signal[:,1].max()-demodulated_signal[:,1].min())
        elif mode == 'optimal':
            raise NotImplementedError('Optimal Demodulation Mode not coded yet.')
        else:
            raise NotImplementedError('Demodulation mode not understood.')

        self.integrated_signal = integrated_signal
        self.demodulated_signal = demodulated_signal.tolist()

        return integrated_signal

    def _reset(self):
        #reset QRM
        self._qrm.reset()

    def stop(self):
	    #stop current sequence running in QRM
        self._qrm.stop_sequencer()

    def __del__(self):
	    #close connection to QRM
        self._qrm.close()

class Pulsar_QCM():

    def __init__(self, label, ip, settings = {}):
        """
        This method connects to the Pulsar QCM and configures it with those settings 
        that are not expected to change.
        All parameters within settings are optional, their default values are:
            QRM_init_settings = {
                    'ref_clock': 'external',                        # Clock source ['external', 'internal']
                    'sequencer': 0,
                    'sync_en': True,
            }
        """        
        settings.setdefault('ref_clock', 'external')
        settings.setdefault('sequencer', 0)
        settings.setdefault('sync_en', True)

        # Instantiate base object from qblox library and connect to it
        qcm = pulsar_qcm(label, ip)

        # Reset and configure
        qcm.reset()
        qcm.reference_source(settings['ref_clock'])

        if settings['sequencer'] == 1:
            qcm.sequencer1_sync_en(settings['sync_en'])
        else:
            qcm.sequencer0_sync_en(settings['sync_en'])

        self._qcm = qcm
        self._settings = settings


    def setup(self, settings = {}):
        settings.setdefault('gain', 0.5)
        settings.setdefault('hardware_avg', 1024)

        settings.setdefault('pulse', {"freq_if": 100e6,
                                        "amplitude": 0.25,
                                        "length": 300,
                                        "offset_i": 0,
                                        "offset_q": 0,
                                        "shape": "Gaussian",
                                        "delay_before": 4,
                                        "repetition_duration": 200000,
                                      })

        self._settings.update(settings)

        qcm = self._qcm
        sequencer = self._settings['sequencer']

        if sequencer == 1:
            qcm.sequencer1_gain_awg_path0(settings['gain'])
            qcm.sequencer1_gain_awg_path1(settings['gain'])
        else:
            qcm.sequencer0_gain_awg_path0(settings['gain'])
            qcm.sequencer0_gain_awg_path1(settings['gain'])

    def set_waveforms(self, waveforms):
        self._waveforms = waveforms
    def set_waveforms_from_pulses_definition(self, pulses_definition: dict):
        for name,pulse in pulses_definition.items():
            pulse_waveforms = generate_waveforms(pulse)
            combined_waveforms = {
                "modI_qcm": {"data": [], "index": 0},
                "modQ_qcm": {"data": [], "index": 1}
            }
            combined_waveforms["modI_qcm"]["data"] = np.concatenate((combined_waveforms["modI_qcm"]["data"],np.zeros(4), pulse_waveforms["modI"]["data"]))
            combined_waveforms["modQ_qcm"]["data"] = np.concatenate((combined_waveforms["modQ_qcm"]["data"],np.zeros(4), pulse_waveforms["modQ"]["data"]))
        self._waveforms = combined_waveforms
        if debugging:
            # Plot the result
            fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
            ax.plot(combined_waveforms["modI_qcm"]["data"],'-',color='C0')
            ax.plot(combined_waveforms["modQ_qcm"]["data"],'-',color='C1')
            ax.title.set_text('Combined Pulses')
    
    def set_program(self, program):
        self._program = program
    def set_program_from_parameters(self, parameters):
        self._program = generate_program(parameters)
    
    def set_acquisitions(self, acquisitions = {"single":   {"num_bins": 1, "index":0}}):
        self._acquisitions = acquisitions
    def set_weights(self, weights = {}):
        self._weights = weights

    def upload_sequence(self):    
        waveforms = self._waveforms
        program = self._program
        acquisitions = self._acquisitions
        weights = self._weights
        qcm = self._qcm
        sequencer = self._settings['sequencer']
        # Upload waveforms and program
        # Reformat waveforms to lists.
        for name in waveforms:
            if str(type(waveforms[name]["data"]).__name__) == "ndarray":
                waveforms[name]["data"] = waveforms[name]["data"].tolist()  # JSON only supports lists

        #Add sequence program and waveforms to single dictionary and write to JSON file.
        wave_and_prog_dict = {"waveforms": waveforms, "weights": weights, "acquisitions": acquisitions, "program": program}
        with open(".data/qcm_sequence.json", 'w', encoding='utf-8') as file:
            json.dump(wave_and_prog_dict, file, indent=4)
            file.close()

        #Upload waveforms and programs.
        if sequencer == 1:
            qcm.sequencer1_waveforms_and_program(os.path.join(os.getcwd(), ".data/qcm_sequence.json"))
        else:
            qcm.sequencer0_waveforms_and_program(os.path.join(os.getcwd(), ".data/qcm_sequence.json"))

        self._wave_and_prog_dict = wave_and_prog_dict



    def play_sequence(self):
        qcm = self._qcm
        settings = self._settings
        sequencer = settings['acq_sequencer']
        #arm sequencer and start playing sequence
        qcm.arm_sequencer()
        qcm.start_sequencer()
        if debugging:
            print(qcm.get_sequencer_state(sequencer))

    def _reset(self):
        #reset QCM
        self._qcm.reset()

    def stop(self):
	    #stop current sequence running in QCM
        self._qcm.stop_sequencer()

    def __del__(self):
	    #close connection to QCM
        self._qcm.close()
