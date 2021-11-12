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

from pulsar_qcm.pulsar_qcm import pulsar_qcm

#import qcodes.instrument_drivers.rohde_schwarz.SGS100A as SGS100A
def calculate_repetition_rate(repetition_duration=200000,
                              wait_loop_step=10000,
                              duration_base=16384):
    extra_duration = repetition_duration-duration_base
    extra_wait = extra_duration%wait_loop_step
    num_wait_loops = (extra_duration-extra_wait)//wait_loop_step
    return num_wait_loops,extra_wait

class Qblox_QCM():
    def __init__(self,info):
        self.info = info
        self.pulsar_qcm = ()

    def connect_qcm_ip(self):
        qcm_ip = self.info["qcm_ip"]
        self.pulsar_qcm = pulsar_qcm("qcm", qcm_ip)

    def reset_qcm(self):
        pulsar_qcm = self.pulsar_qcm
        pulsar_qcm.reset()
        #print(pulsar_qcm.get_system_status())


    def set_waveforms_power_scan(self,pars):
        #Waveform parameters
        wf_qcm_1 = self.info["waveform_qcm_1"]
        freq_if = wf_qcm_1["IF_frequency"]
        leng = wf_qcm_1["waveform_length"]
        offset_i = wf_qcm_1["offset_i"]
        offset_q = wf_qcm_1["offset_q"]

        amp = pars.qcm_amp()

        self.freq_if = freq_if
        self.leng = leng
        self.offset_i = offset_i
        self.offset_q = offset_q
        self.amp = amp

    def set_waveforms_length_scan(self,pars):
        #Waveform parameters
        wf_qcm_1 = self.info["waveform_qcm_1"]
        freq_if = wf_qcm_1["IF_frequency"]
        offset_i = wf_qcm_1["offset_i"]
        offset_q = wf_qcm_1["offset_q"]
        amp = wf_qcm_1["amplitude"]  

        leng = int(pars.qcm_leng())

        self.freq_if = freq_if
        self.leng = leng
        self.offset_i = offset_i
        self.offset_q = offset_q
        self.amp = amp

    def modulate_envolope(self):
        wf_qcm_1 = self.info["waveform_qcm_1"]
        freq_if = self.freq_if
        leng = self.leng
        offset_i = self.offset_i
        offset_q = self.offset_q
        wf_type = wf_qcm_1["waveform_type"]
        std = leng/5
        amp = self.amp
        if wf_type == 'Block':
            envelope_i = amp*np.ones(int(leng))
            envelope_q = amp*np.zeros(int(leng))
        elif wf_type == 'Gaussian':
            envelope_i = amp*scipy.signal.gaussian(leng, std=std)
            envelope_q = amp*np.zeros(int(leng))#amp*scipy.signal.gaussian(leng, std=std)
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
                "modI_qcm_1": {"data": [], "index": 0},
                "modQ_qcm_1": {"data": [], "index": 1}
            }

        # adding mixer offsets
        waveforms["modI_qcm_1"]["data"] = mod_signals[:,0]+offset_i
        waveforms["modQ_qcm_1"]["data"] = mod_signals[:,1]+offset_q
        self.waveforms = waveforms 

    def modulate_envolope_echo(self):
        wf_qcm_1 = self.info["waveform_qcm_1"]
        freq_if = self.freq_if
        leng = self.leng
        amp = self.amp
        offset_i = self.offset_i
        offset_q = self.offset_q
        wf_type = wf_qcm_1["waveform_type"]
        std = leng/5
          

        if wf_type == 'Block':
            envelope_i = amp*np.ones(int(leng))
            envelope_q = amp*np.zeros(int(leng))
        elif wf_type == 'Gaussian':
            envelope_i = amp*scipy.signal.gaussian(leng, std=std)
            envelope_q = amp*np.zeros(int(leng))#amp*scipy.signal.gaussian(leng, std=std)
            envelope_i_pi = 2*amp*scipy.signal.gaussian(leng, std=std)
            envelope_q_pi = 2*amp*np.zeros(int(leng))#amp*scipy.signal.gaussian(leng, std=std)

        time = np.arange(envelope_i.shape[0])*1e-9
        cosalpha = np.cos(2*np.pi*freq_if*time)
        sinalpha = np.sin(2*np.pi*freq_if*time)
        mod_matrix = np.array([[cosalpha,sinalpha],[-sinalpha,cosalpha]])
        result = []
        for it,t,ii,qq in zip(np.arange(envelope_i.shape[0]),time,envelope_i,envelope_q):
            result.append(mod_matrix[:,:,it]@np.array([ii,qq]))
        mod_signals = np.array(result)
        result = []
        for it,t,ii,qq in zip(np.arange(envelope_i_pi.shape[0]),time,envelope_i_pi,envelope_q_pi):
            result.append(mod_matrix[:,:,it]@np.array([ii,qq]))
        mod_signals_pi = np.array(result)
        #Waveform dictionary (data will hold the sampples and index will be used to select the waveforms in the instrumment).
        waveforms = {
                "modI_qcm_1": {"data": [], "index": 0},
                "modQ_qcm_1": {"data": [], "index": 1},
                "modI_qcm_1_2": {"data": [], "index": 2},
                "modQ_qcm_1_2": {"data": [], "index": 3}
            }

        # adding mixer offsets
        waveforms["modI_qcm_1"]["data"] = mod_signals[:,0]+offset_i
        waveforms["modQ_qcm_1"]["data"] = mod_signals[:,1]+offset_q
        waveforms["modI_qcm_1_2"]["data"] = mod_signals_pi[:,0]+offset_i
        waveforms["modQ_qcm_1_2"]["data"] = mod_signals_pi[:,1]+offset_q
        self.waveforms = waveforms 

    def set_waveforms(self):
        #Waveform parameters
        wf_qcm_1 = self.info["waveform_qcm_1"]
        freq_if = wf_qcm_1["IF_frequency"]
        amp = wf_qcm_1["amplitude"]    
        leng = wf_qcm_1["waveform_length"]
        offset_i = wf_qcm_1["offset_i"]
        offset_q = wf_qcm_1["offset_q"]
        wf_type = wf_qcm_1["waveform_type"]
        std = leng/5
        # modulate_envolope
        print(f'parsed amp = {amp} V')
        if wf_type == 'Block':
            envelope_i = amp*np.ones(leng)
            envelope_q = amp*np.ones(leng)
        elif wf_type == 'Gaussian':
            envelope_i = amp*scipy.signal.gaussian(leng, std=std)
            envelope_q = np.zeros(leng)#amp*scipy.signal.gaussian(leng, std=std)
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
                "modI_qcm_1": {"data": [], "index": 0},
                "modQ_qcm_1": {"data": [], "index": 1}
            }

        # adding mixer offsets
        waveforms["modI_qcm_1"]["data"] = mod_signals[:,0]+offset_i
        waveforms["modQ_qcm_1"]["data"] = mod_signals[:,1]+offset_q
        self.waveforms = waveforms 
        self.freq_if = freq_if
        self.leng = leng
        self.offset_i = offset_i
        self.offset_q = offset_q
        self.amp = amp

        '''
        fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
        ax.plot(self.waveforms["modI_qcm_1"]["data"],'-',color='C0')
        ax.plot(self.waveforms["modQ_qcm_1"]["data"],'-',color='C1')
        ax.title.set_text('QCM_Ch1&2_output')
        '''

    def set_waveforms_for_MC(self,pars):
        #Waveform parameters
        wf_qcm_1 = self.info["waveform_qcm_1"]
        pars.freq_if(wf_qcm_1["IF_frequency"])
        #pars.amp(wf_qcm_1["amplitude"])
        pars.leng(wf_qcm_1["waveform_length"])
        freq_if = pars.freq_if()
        amp = pars.amp() 
        leng = pars.leng()
        offset_i = wf_qcm_1["offset_i"]
        offset_q = wf_qcm_1["offset_q"]
        wf_type = wf_qcm_1["waveform_type"]
        std = leng/5
        # modulate_envolope
        if wf_type == 'Block':
            envelope_i = amp*np.ones(leng)
            envelope_q = amp*np.zeros(leng)
        elif wf_type == 'Gaussian':
            envelope_i = amp*scipy.signal.gaussian(leng, std=std)
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
                "modI_qcm_1": {"data": [], "index": 0},
                "modQ_qcm_1": {"data": [], "index": 1}
            }

        # adding mixer offsets
        waveforms["modI_qcm_1"]["data"] = mod_signals[:,0]+offset_i
        waveforms["modQ_qcm_1"]["data"] = mod_signals[:,1]+offset_q
        self.waveforms = waveforms

    def set_waveforms_rabi_waveform_leng(self):
        #Waveform parameters
        wf_qcm_1 = self.info["waveform_qcm_1"]
        freq_if = wf_qcm_1["IF_frequency"]
        amp = wf_qcm_1["amplitude"]    
        leng = self.leng
        offset_i = wf_qcm_1["offset_i"]
        offset_q = wf_qcm_1["offset_q"]
        wf_type = wf_qcm_1["waveform_type"]
        std = leng/5
        # modulate_envolope
        print(f'parsed amp = {amp} V')
        if wf_type == 'Block':
            envelope_i = amp*np.ones(leng)
            envelope_q = amp*np.zeros(leng)
        elif wf_type == 'Gaussian':
            envelope_i = amp*scipy.signal.gaussian(leng, std=std)
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
                "modI_qcm_1": {"data": [], "index": 0},
                "modQ_qcm_1": {"data": [], "index": 1}
            }

        # adding mixer offsets
        waveforms["modI_qcm_1"]["data"] = mod_signals[:,0]+offset_i
        waveforms["modQ_qcm_1"]["data"] = mod_signals[:,1]+offset_q
        self.waveforms = waveforms 
        self.freq_if = freq_if
        self.leng = leng
        self.offset_i = offset_i
        self.offset_q = offset_q
        self.amp = amp




    def sequence_program_qubit_spec(self,repetition_duration=200000):
        wait_loop_step=1000
        num_wait_loops,extra_wait = calculate_repetition_rate(repetition_duration=repetition_duration,
                                                              wait_loop_step=wait_loop_step,
                                                              duration_base=16384)
        seq_prog = f"""
            move    {self.info["number_of_average"]},R0
            nop
            wait_sync 4           # Synchronize sequencers over multiple instruments
            
        loop:
            play      0,1,4   # Play waveforms (0,1) and wait remaining duration of scope acquisition.
            wait      16380
            move      {num_wait_loops},R1      # repetion rate loop iterator
            nop
            reprateloop:
                wait      {wait_loop_step}
                loop      R1,@reprateloop
            wait      {extra_wait}
            loop      R0,@loop

            stop
        """

        self.seq_prog = seq_prog

    def sequence_program_echo(self,wait_time_ns=20,repetition_duration=200000):
        wait_loop_step=1000
        num_wait_loops,extra_wait = calculate_repetition_rate(repetition_duration=repetition_duration-wait_time_ns,
                                                              wait_loop_step=wait_loop_step,
                                                              duration_base=16384)

        seq_prog = f"""
            move    {self.info["number_of_average"]},R0
            nop
            wait_sync 4           # Synchronize sequencers over multiple instruments
            
        loop:
            play      0,1,4   # Play waveforms (0,1) and wait remaining duration of scope acquisition.
            wait      {int(self.leng+wait_time_ns/2)}
            play      2,3,4   # Play waveforms (2,3) and wait remaining duration of scope acquisition.
            wait      {int(self.leng+wait_time_ns/2)}
            play      0,1,4   # Play waveforms (0,1) and wait remaining duration of scope acquisition.

            wait      {16384-4-4-4-2*self.leng}
            move      {num_wait_loops},R1      # repetion rate loop iterator
            nop
            reprateloop:
                wait      {wait_loop_step}
                loop      R1,@reprateloop
            wait      {extra_wait}
            loop      R0,@loop

            stop
        """

        self.seq_prog = seq_prog


    def sequence_program_ramsey(self,wait_time_ns=20,repetition_duration=200000):
        wait_loop_step=1000
        num_wait_loops,extra_wait = calculate_repetition_rate(repetition_duration=repetition_duration-wait_time_ns,
                                                              wait_loop_step=wait_loop_step,
                                                              duration_base=16384)
        seq_prog = f"""
            move    {self.info["number_of_average"]},R0
            nop
            wait_sync 4           # Synchronize sequencers over multiple instruments
            
        loop:
            play      0,1,4   # Play waveforms (0,1) and wait remaining duration of scope acquisition.
            wait      {self.leng+wait_time_ns}
            play      0,1,4   # Play waveforms (0,1) and wait remaining duration of scope acquisition.
            wait      {16384-4-4-self.leng}
            move      {num_wait_loops},R1      # repetion rate loop iterator
            nop
            reprateloop:
                wait      {wait_loop_step}
                loop      R1,@reprateloop
            wait      {extra_wait}
            loop      R0,@loop

            stop
        """

        self.seq_prog = seq_prog


    def upload_waveforms(self,acquisitions_qrm):
        waveforms = self.waveforms
        pulsar_qcm = self.pulsar_qcm
        seq_prog = self.seq_prog
        # Reformat waveforms to lists if necessary.
        for name in waveforms:
            if str(type(waveforms[name]["data"]).__name__) == "ndarray":
                waveforms[name]["data"] = waveforms[name]["data"].tolist()  # JSON only supports lists

        #Add sequence program and waveforms to single dictionary and write to JSON file.
        wave_and_prog_dict = {"waveforms": waveforms, "weights":{}, "acquisitions": acquisitions_qrm, "program": seq_prog}
        with open("qcm_sequence.json", 'w', encoding='utf-8') as file:
            json.dump(wave_and_prog_dict, file, indent=4)
            file.close()

        pulsar_qcm.sequencer0_waveforms_and_program(os.path.join(os.getcwd(), "qcm_sequence.json"))

        self.wave_and_prog_dict = wave_and_prog_dict

    def play_continuous(self):
        pulsar_qcm = self.pulsar_qcm
        pulsar_qcm.set("sequencer0_cont_mode_en_awg_path0".format(0),True)
        pulsar_qcm.set("sequencer0_cont_mode_en_awg_path1".format(1),True)
        pulsar_qcm.sequencer0_cont_mode_waveform_idx_awg_path0(0)
        pulsar_qcm.sequencer0_cont_mode_waveform_idx_awg_path1(1)

    def set_sync_and_gain_of_sequencer(self):
        pulsar_qcm = self.pulsar_qcm
        pulsar_qcm.sequencer0_sync_en(True)

    def arm_and_start_sequencer(self):
        pulsar_qcm = self.pulsar_qcm
        #Arm sequencer.
        pulsar_qcm.arm_sequencer()
        #Start sdequencer.
        pulsar_qcm.start_sequencer()
        #Print status of the QRM (only sequencer 0)
        #print("QCM status:")
        #print(pulsar_qcm.get_sequencer_state(0))
        #print()
        self.status_start_seqencer0 = pulsar_qcm.get_sequencer_state(0)

    def wait_sequencer_to_stop(self):
        #Wait for the sequencer to stop with a timeout period of one minute.
        self.pulsar_qcm.get_sequencer_state(0,1)

    def stop_sequencers(self):
        self.pulsar_qcm.stop_sequencer()

    def close_connections(self):
        self.pulsar_qcm.close()




