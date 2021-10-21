#Set up the environment. 
import os 
import scipy.signal
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import lmfit
import xarray as xr

from qcodes import ManualParameter, Parameter

from pathlib import Path
from quantify.data.handling import get_datadir, set_datadir
from quantify.measurement import MeasurementControl
from quantify.measurement.control import Settable, Gettable
import quantify.visualization.pyqt_plotmon as pqm
from quantify.visualization.instrument_monitor import InstrumentMonitor

from pulsar_qrm.pulsar_qrm import pulsar_qrm

import qcodes.instrument_drivers.rohde_schwarz.SGS100A as SGS100A

def calculate_repetition_rate(repetition_duration=200000,
                              wait_loop_step=10000,
                              duration_base=16384):
    extra_duration = repetition_duration-duration_base
    extra_wait = extra_duration%wait_loop_step
    num_wait_loops = (extra_duration-extra_wait)//wait_loop_step
    return num_wait_loops,extra_wait

class Qblox_QRM():
    def __init__(self,info):
        self.info = info
        self.pulsar_qrm = ()

    def set_data_dictionary(self):
        data_dict = self.info["data_dictionary"]
        set_datadir(data_dict) 
        print(f"Data will be saved in:\n{get_datadir()}")
        return get_datadir()
        
    def connect_qrm_ip(self):
        #Connect to the Pulsar QRM at IP address.
        qrm_ip = self.info["qrm_ip"]
        self.pulsar_qrm = pulsar_qrm("qrm", qrm_ip)
        #self.pulsar_qrm = pulsar_qrm
        
    def reset_qrm(self):
        #Reset the instrument for good measure.
        pulsar_qrm = self.pulsar_qrm
        pulsar_qrm.reset()
        #print(pulsar_qrm.get_system_status())

    def reference_clock_external(self):
        pulsar_qrm = self.pulsar_qrm
        pulsar_qrm.reference_source("external")

    def prepare_plot_windows(self):
        MC = MeasurementControl('MC')
        # Create the live plotting intrument which handles the graphical interface
        # Two windows will be created, the main will feature 1D plots and any 2D plots will go to the secondary
        plotmon = pqm.PlotMonitor_pyqt('plotmon')
        # Connect the live plotting monitor to the measurement control
        MC.instr_plotmon(plotmon.name)
        # The instrument monitor will give an overview of all parameters of all instruments
        insmon = InstrumentMonitor("Instruments Monitor")
        # By connecting to the MC the parameters will be updated in real-time during an experiment.
        MC.instrument_monitor(insmon.name)

        self.MC = MC
        self.plotmon = plotmon 
        self.insmon = insmon

    def set_waveforms(self):
        #Waveform parameters
        wf_qrm = self.info["waveform_qrm"]
        freq_if = wf_qrm["IF_frequency"]
        amp = wf_qrm["amplitude"]
        leng = wf_qrm["waveform_length"]
        offset_i = wf_qrm["offset_i"]
        offset_q = wf_qrm["offset_q"]

        self.freq_if = freq_if
        self.leng = leng
        self.offset_i = offset_i
        self.offset_q = offset_q
        self.amp = amp

    def set_waveforms_punchout(self,pars):
        #Waveform parameters
        wf_qrm = self.info["waveform_qrm"]
        freq_if = wf_qrm["IF_frequency"]
        leng = wf_qrm["waveform_length"]
        offset_i = wf_qrm["offset_i"]
        offset_q = wf_qrm["offset_q"]

        amp = pars.qrm_amp()

        self.freq_if = freq_if
        self.leng = leng
        self.offset_i = offset_i
        self.offset_q = offset_q
        self.amp = amp

    def modulate_envolope(self):
        freq_if = self.freq_if
        leng = self.leng
        offset_i = self.offset_i
        offset_q = self.offset_q
        amp = self.amp
        envelope_i = amp*np.ones(leng)
        envelope_q = amp*np.zeros(leng)
        time = np.arange(envelope_i.shape[0])*1e-9
        cosalpha = np.cos(2*np.pi*freq_if*time)
        sinalpha = np.sin(2*np.pi*freq_if*time)
        mod_matrix = np.array([[cosalpha,sinalpha],[-sinalpha,cosalpha]])
        result = []
        for it,t,ii,qq in zip(np.arange(envelope_i.shape[0]),time,envelope_i,envelope_q):
            result.append(mod_matrix[:,:,it]@np.array([ii,qq]))
        mod_signals = np.array(result)
        #Waveform dictionary (data will hold the sampples and index will be used to select the waveforms in the instrumment).
        waveforms = {
                "modI_qrm": {"data": [], "index": 0},
                "modQ_qrm": {"data": [], "index": 1}
            }

        # adding mixer offsets
        waveforms["modI_qrm"]["data"] = mod_signals[:,0]+offset_i
        waveforms["modQ_qrm"]["data"] = mod_signals[:,1]+offset_q
        self.waveforms = waveforms


        '''
        fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
        ax.plot(self.waveforms["modI_qrm"]["data"],'-',color='C0')
        ax.plot(self.waveforms["modQ_qrm"]["data"],'-',color='C1')
        ax.title.set_text('QRM output')
        '''

    def specify_acquisitions(self):

        acquisitions = {"single":   {"num_bins": 1,
                                     "index":0}}
        self.acquisitions = acquisitions

    def sequence_program_single(self):
        seq_prog = """
        play    0,1,4     # Play waveforms (0,1) in channels (O0,O1) and wait 4ns.
        acquire 0,0,16380 # Acquire waveforms over remaining duration of acquisition of input vector of length = 16380 with integration weights 0,0
        stop              # Stop.
        """
        self.seq_prog = seq_prog

    def sequence_program_average(self):
        seq_prog = f"""
            move    {self.info["number_of_average"]},R0
            nop
        loop:
            play    0,1,4
            acquire 0,0,16380
            loop    R0,@loop

            stop
        """
        self.seq_prog = seq_prog

    def sequence_program_qubit_spec(self,qcm_leng,repetition_duration=200000):
        wait_loop_step=1000
        num_wait_loops,extra_wait = calculate_repetition_rate(repetition_duration=repetition_duration,
                                                              wait_loop_step=wait_loop_step,
                                                              duration_base=16384)
        buffer_time = 40  #ns


        seq_prog = f"""
            move    {self.info["number_of_average"]},R0
            nop
            wait_sync 4           # Synchronize sequencers over multiple instruments

        loop:
            wait      {qcm_leng+buffer_time} # idle for xx ns gaussian pulse + 40 ns buffer
            play      0,1,4      # Play waveforms (0,1) in channels (O0,O1) and wait 4ns.
            acquire   0,0,4      # Acquire waveforms over remaining duration of acquisition of input vector of length = 16380 with integration weights 0,0
            wait      {16380-4-qcm_leng-buffer_time}
            move      {num_wait_loops},R1      # repetion rate loop iterator
            nop
            reprateloop:
                wait      {wait_loop_step}
                loop      R1,@reprateloop
            wait      {extra_wait}
            loop    R0,@loop

            stop
        """


        self.seq_prog = seq_prog


    def sequence_program_t1(self,qcm_leng,wait_time_ns=20,repetition_duration=200000):
        wait_loop_step=1000
        num_wait_loops,extra_wait = calculate_repetition_rate(repetition_duration=repetition_duration-wait_time_ns,
                                                              wait_loop_step=wait_loop_step,
                                                              duration_base=16384)
        buffer_time = 40  #ns


        seq_prog = f"""
            move    {self.info["number_of_average"]},R0
            nop
            wait_sync 4           # Synchronize sequencers over multiple instruments

        loop:
            wait      {qcm_leng+buffer_time} # idle for xx ns gaussian pulse + 40 ns buffer
            wait      {wait_time_ns}
            play      0,1,4      # Play waveforms (0,1) in channels (O0,O1) and wait 4ns.
            acquire   0,0,4      # Acquire waveforms over remaining duration of acquisition of input vector of length = 16380 with integration weights 0,0
            wait      {16380-4-qcm_leng-buffer_time}
            move      {num_wait_loops},R1      # repetion rate loop iterator
            nop
            reprateloop:
                wait      {wait_loop_step}
                loop      R1,@reprateloop
            wait      {extra_wait}
            loop    R0,@loop

            stop
        """


        self.seq_prog = seq_prog

    def sequence_program_echo(self,qcm_leng,wait_time_ns=20,repetition_duration=200000):
        wait_loop_step=1000
        num_wait_loops,extra_wait = calculate_repetition_rate(repetition_duration=repetition_duration-wait_time_ns,
                                                              wait_loop_step=wait_loop_step,
                                                              duration_base=16384)
        buffer_time = 40  #ns




        seq_prog = f"""
            move    {self.info["number_of_average"]},R0
            nop
            wait_sync 4           # Synchronize sequencers over multiple instruments

        loop:
            wait      {3*qcm_leng+wait_time_ns+2*qcm_leng+buffer_time} # idle for xx ns gaussian pulse + 40 ns buffer
            play      0,1,4      # Play waveforms (0,1) in channels (O0,O1) and wait 4ns.
            acquire   0,0,4      # Acquire waveforms over remaining duration of acquisition of input vector of length = 16380 with integration weights 0,0
            wait      {16384-4-4-3*qcm_leng-2*qcm_leng-buffer_time}
            move      {num_wait_loops},R1      # repetion rate loop iterator
            nop
            reprateloop:
                wait      {wait_loop_step}
                loop      R1,@reprateloop
            wait      {extra_wait}
            loop    R0,@loop

            stop
        """

        self.seq_prog = seq_prog

    def sequence_program_ramsey(self,qcm_leng,wait_time_ns=20,repetition_duration=200000):
        wait_loop_step=1000
        num_wait_loops,extra_wait = calculate_repetition_rate(repetition_duration=repetition_duration-wait_time_ns,
                                                              wait_loop_step=wait_loop_step,
                                                              duration_base=16384)
        buffer_time = 40  #ns


        seq_prog = f"""
            move    {self.info["number_of_average"]},R0
            nop
            wait_sync 4           # Synchronize sequencers over multiple instruments

        loop:
            wait      {qcm_leng*3+wait_time_ns+buffer_time} # idle for xx ns gaussian pulse + 40 ns buffer
            play      0,1,4      # Play waveforms (0,1) in channels (O0,O1) and wait 4ns.
            acquire   0,0,4      # Acquire waveforms over remaining duration of acquisition of input vector of length = 16380 with integration weights 0,0
            wait      {16384-4-4-3*qcm_leng-buffer_time}
            move      {num_wait_loops},R1      # repetion rate loop iterator
            nop
            reprateloop:
                wait      {wait_loop_step}
                loop      R1,@reprateloop
            wait      {extra_wait}
            loop    R0,@loop

            stop
        """


        self.seq_prog = seq_prog


    def upload_waveforms(self):
        waveforms = self.waveforms
        pulsar_qrm = self.pulsar_qrm
        seq_prog = self.seq_prog
        acquisitions = self.acquisitions
        # Reformat waveforms to lists if necessary.
        for name in waveforms:
            if str(type(waveforms[name]["data"]).__name__) == "ndarray":
                waveforms[name]["data"] = waveforms[name]["data"].tolist()  # JSON only supports lists

        #Add sequence program and waveforms to single dictionary and write to JSON file.
        wave_and_prog_dict = {"waveforms": waveforms, "weights":{}, "acquisitions": acquisitions, "program": seq_prog}
        with open("qrm_sequence.json", 'w', encoding='utf-8') as file:
            json.dump(wave_and_prog_dict, file, indent=4)
            file.close()

         #Upload waveforms and programs.
        pulsar_qrm.sequencer0_waveforms_and_program(os.path.join(os.getcwd(), "qrm_sequence.json"))
        self.wave_and_prog_dict = wave_and_prog_dict
    '''
    def upload_waveforms_from_qcm(self,waveforms_qcm):
        pulsar_qrm = self.pulsar_qrm
        seq_prog = self.seq_prog
        acquisitions = self.acquisitions
        waveforms = self.waveforms
        # Reformat waveforms from qcm to lists if necessary.
        for name in waveforms_qcm:
            if str(type(waveforms_qcm[name]["data"]).__name__) == "ndarray":
                waveforms_qcm[name]["data"] = waveforms_qcm[name]["data"].tolist()  # JSON only supports lists
    
        # Reformat waveforms from qrm to lists if necessary.
        for name in waveforms:
            waveforms[name]["index"] = waveforms[name]["index"]+2
            if str(type(waveforms[name]["data"]).__name__) == "ndarray":
                waveforms[name]["data"] = waveforms[name]["data"].tolist()  # JSON only supports lists

        #Add sequence program and waveforms to single dictionary and write to JSON file.
        wave_and_prog_dict = {"waveforms": dict(list(waveforms_qcm.items()) + list(waveforms.items())), "weights":{}, "acquisitions": acquisitions, "program": seq_prog}
        with open("qrm_sequence.json", 'w', encoding='utf-8') as file:
            json.dump(wave_and_prog_dict, file, indent=4)
            file.close()

        #Upload waveforms and programs.
        pulsar_qrm.sequencer0_waveforms_and_program(os.path.join(os.getcwd(), "qrm_sequence.json"))

        self.waveforms_qcm = waveforms_qcm
        self.wave_and_prog_dict = wave_and_prog_dict
    '''

    def configure_scope_acquisition(self):
        pulsar_qrm = self.pulsar_qrm
        pulsar_qrm.scope_acq_sequencer_select(0) 
        pulsar_qrm.scope_acq_trigger_mode_path0("sequencer")
        pulsar_qrm.scope_acq_trigger_mode_path1("sequencer")

    def enable_hardware_averaging(self):
        pulsar_qrm =self.pulsar_qrm
        pulsar_qrm.scope_acq_sequencer_select(0) 
        pulsar_qrm.scope_acq_avg_mode_en_path0(True)
        pulsar_qrm.scope_acq_avg_mode_en_path1(True)

    def configure_sequencer_sync(self):
        pulsar_qrm = self.pulsar_qrm
        pulsar_qrm.sequencer0_sync_en(True)

    def arm_and_start_sequencer(self):
        pulsar_qrm = self.pulsar_qrm
        #Arm sequencer.
        pulsar_qrm.arm_sequencer()
        #Start sdequencer.
        pulsar_qrm.start_sequencer()
        #Print status of the QRM (only sequencer 0)
        #print("QRM status:")
        #print(pulsar_qrm.get_sequencer_state(0))
        #print()
        self.status_start_seqencer0 = pulsar_qrm.get_sequencer_state(0)

    def acquisition(self):
        pulsar_qrm = self.pulsar_qrm
        #Wait for the sequencer to stop with a timeout period of one minute.
        pulsar_qrm.get_sequencer_state(0, 1)

        #Wait for the acquisition to finish with a timeout period of one second.
        pulsar_qrm.get_acquisition_state(0, 1)

        #Move acquisition data from temporary memory to acquisition list.
        pulsar_qrm.store_scope_acquisition(0, "single")

        #Get acquisition list from instrument.
        single_acq = pulsar_qrm.get_acquisitions(0)

        self.single_acq = single_acq

    def plot_acquisitions(self):
        #Plot acquired signal on both inputs.
        fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
        ax.plot(self.single_acq["single"]["acquisition"]["scope"]["path0"]["data"][120:6500])
        ax.plot(self.single_acq["single"]["acquisition"]["scope"]["path1"]["data"][120:6500])
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Relative amplitude')
        plt.show()

    def stop_sequencers(self):
        self.pulsar_qrm.stop_sequencer()

    def close_connections(self):
        self.pulsar_qrm.close()

    def demodulate_and_integrate(self):

        wf_qrm = self.info["waveform_qrm"]
        freq_if = wf_qrm["IF_frequency"]
        leng = wf_qrm["waveform_length"]

        inte_info = self.info["integral_info"]
        start_sample = inte_info["start_sample"]
        sampling_rate = inte_info["sampling_rate"]
        integration_length = inte_info["integration_length"]
        mode = inte_info["mode"]

        norm_factor = 1./(integration_length)
        input_vec_I = np.array(self.single_acq["single"]["acquisition"]["scope"]["path0"]["data"][start_sample:start_sample+integration_length])
        input_vec_Q = np.array(self.single_acq["single"]["acquisition"]["scope"]["path1"]["data"][start_sample:start_sample+integration_length])
        input_vec_I -= np.mean(input_vec_I)
        input_vec_Q -= np.mean(input_vec_Q)
        
        """
        fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
        ax.plot(self.single_acq["single"]["acquisition"]["scope"]["path0"]["data"][0:1000])
        ax.plot(self.single_acq["single"]["acquisition"]["scope"]["path1"]["data"][0:1000])

        print(np.max(self.single_acq["single"]["acquisition"]["scope"]["path0"]["data"][0:1000])-np.min(self.single_acq["single"]["acquisition"]["scope"]["path0"]["data"][0:1000]))
        print(np.max(self.single_acq["single"]["acquisition"]["scope"]["path1"]["data"][0:1000])-np.min(self.single_acq["single"]["acquisition"]["scope"]["path1"]["data"][0:1000]))
        ax.set_xlabel('Time (ns)')
        ax.set_xlim(100,150)
        #ax.set_ylim(-0.01,-0.007)
        ax.set_ylabel('Relative amplitude')
        #ax.twinx().plot(self.pulsar_qrm.get_waveforms(0)['modI_qrm']['data'],color='r')
        plt.show()
        """
        """
        fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
        data = self.single_acq["single"]["acquisition"]["scope"]["path1"]["data"][start_sample:start_sample+integration_length]
        fft_data = np.fft.fft(np.array(data)-np.mean(data))
        fft_freq = np.fft.fftfreq(len(data), d=1e-9)
        mask_sort = np.argsort(fft_freq)
        ax.plot(fft_freq[mask_sort],fft_data[mask_sort])
        ax.set_xlabel('Time (ns)')
        ax.set_xlim(-150e6,150e6)
        ax.set_ylabel('Relative amplitude')
        plt.show()
        """
        if mode=='ssb':
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
            #print(integrated_signal,demodulated_signal[:,0].max()-demodulated_signal[:,0].min(),demodulated_signal[:,1].max()-demodulated_signal[:,1].min())
            self.integrated_signal = integrated_signal
            self.demodulated_signal = demodulated_signal.tolist()
            
            """
            fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
            ax.plot(demodulated_signal[:500,0])
            ax.plot(demodulated_signal[:500,1])
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('Relative amplitude')
            ax.set_ylim(0,demodulated_signal.max())
            plt.show()
            """
            
        elif mode=='optimal':
            raise NotImplementedError('Optimal Demodulation Mode not coded yet.')
        else:
            raise NotImplementedError('Demodulation mode not understood.')
        return integrated_signal 
