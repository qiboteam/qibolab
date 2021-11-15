"""
class for interfacing with Qblox QRM and QCM
"""
import os
import scipy.signal
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import lmfit
import xarray as xr

#from qcodes import ManualParameter, Parameter
#from pathlib import Path
from quantify_core.data.handling import get_datadir, set_datadir
#from quantify.measurement import MeasurementControl
#from quantify.measurement.control import Settable, Gettable
#import quantify.visualization.pyqt_plotmon as pqm
#from quantify.visualization.instrument_monitor import InstrumentMonitor

from pulsar_qcm.pulsar_qcm import pulsar_qcm
from pulsar_qrm.pulsar_qrm import pulsar_qrm


class Pulsar_QRM():

	# Construction method
    def __init__(self, label, ip):
        """
        create Qblox QRM with name = label and connect to it in local IP = ip and set reference clock source
        Params format example:
                "ip": '192.168.0.2' (only 192.168.0.X accepted by Qblox)
                "label": "qcm"
        """
        self.label = label
        self.ip = ip
        self.qrm = pulsar_qrm(label, ip)

    #QRM Configuration method
    def setup(self, QRM_info: dict):
        '''
        Function for setting up the Qblox QRM parameters
        Params example:
            QRM_info =
            {
                "data_dictionary": "quantify-data/",
                "ref_clock": external,
                "start_sample": 130,
                'hardware_avg': 1024,
                "integration_length": 600,
                "sampling_rate": 1e9,
                "mode": "ssb"
            }
        '''
        #Set up instrument integration and modulation parameters
        self.start_sample = QRM_info['start_sample']
        self.hardware_avg = QRM_info['hardware_avg']
        self.integration_length = QRM_info['integration_length']
        self.sampling_rate = QRM_info['sampling_rate']
        self.mode = QRM_info['mode']

        self._reset() #reset instrument from previous state
        self._set_reference_clock(QRM_info['ref_clock']) #set reference clock source
        self.set_data_dictionary(QRM_info['data_dictionary']) #set data directory for generated waveforms
        self.specify_acquisitions() #specify acquisitions params in QRM
        self.enable_hardware_averaging() #specify QRM HW averaging
        self.configure_sequencer_sync() #enable sequencer sync

    #Modifiers
    def _set_reference_clock(self, ref_clock):
        #set external reference clock QRM
        self.qrm.reference_source(ref_clock)

    def set_data_dictionary(self, data_dict):
        set_datadir(data_dict)
        print(f"Data will be saved in:\n{get_datadir()}")
        return get_datadir()

    def specify_acquisitions(self):
        #define type Qblox QRM acquisitions. See Qblox documentation to understand format
        acquisitions = {"single":   {"num_bins": 1, "index":0}}
        self.acquisitions = acquisitions

    def enable_hardware_averaging(self):
        #Enable QRM acquisition average mode
        self.qrm.scope_acq_sequencer_select(0)
        self.qrm.scope_acq_avg_mode_en_path0(True)
        self.qrm.scope_acq_avg_mode_en_path1(True)

    def configure_sequencer_sync(self):
        self.qrm.sequencer0_sync_en(True)

    def set_waveforms(self, IF_frequency, waveform_length, offset_i, offset_q, amplitude):
        #Setting waveform parameters
        self.freq_if = IF_frequency
        self.leng = waveform_length
        self.offset_i = offset_i
        self.offset_q = offset_q
        self.amp = amplitude

    def modulate_envolope(self):
        #Waveform UP coversion
        envelope_i = self.amp*np.ones(self.leng)
        envelope_q = self.amp*np.zeros(self.leng)
        time = np.arange(envelope_i.shape[0])*1e-9
        cosalpha = np.cos(2*np.pi*self.freq_if*time)
        sinalpha = np.sin(2*np.pi*self.freq_if*time)
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
        waveforms["modI_qrm"]["data"] = mod_signals[:,0]+self.offset_i
        waveforms["modQ_qrm"]["data"] = mod_signals[:,1]+self.offset_q
        self.waveforms = waveforms

    def set_sequence(self, seq):
        #set Q1ASM sequence to be executed in the QRM
        self.seq_prog = seq

    def upload_waveforms(self, waveforms):
        # Reformat waveforms to lists if necessary.
        for name in waveforms:
            if str(type(self.waveforms[name]["data"]).__name__) == "ndarray":
                self.waveforms[name]["data"] = self.waveforms[name]["data"].tolist()  # JSON only supports lists

        #Add sequence program and waveforms to single dictionary and write to JSON file.
        wave_and_prog_dict = {"waveforms": self.waveforms, "weights":{}, "acquisitions": self.acquisitions, "program": self.seq_prog}
        with open("qrm_sequence.json", 'w', encoding='utf-8') as file:
            json.dump(wave_and_prog_dict, file, indent=4)
            file.close()

        #Upload waveforms and programs.
        self.qrm.sequencer0_waveforms_and_program(os.path.join(os.getcwd(), "qrm_sequence.json"))
        self.wave_and_prog_dict = wave_and_prog_dict

    def arm_and_start_sequencer(self):
        #arm sequencer and start playing sequence
        self.qrm.arm_sequencer()
        self.qrm.start_sequencer()
        self.status_start_seqencer0 = self.qrm.get_sequencer_state(0)

    def acquisition(self):
        #start acquisition of data
        #Wait for the sequencer to stop with a timeout period of one minute.
        self.qrm.get_sequencer_state(0, 1)
        #Wait for the acquisition to finish with a timeout period of one second.
        self.qrm.get_acquisition_state(0, 1)
        #Move acquisition data from temporary memory to acquisition list.
        self.qrm.store_scope_acquisition(0, "single")
        #Get acquisition list from instrument.
        self.single_acq = pulsar_qrm.get_acquisitions(0)

    def plot_acquisitions(self):
        #Plot acquired signal on both inputs.
        fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
        ax.plot(self.single_acq["single"]["acquisition"]["scope"]["path0"]["data"][120:6500])
        ax.plot(self.single_acq["single"]["acquisition"]["scope"]["path1"]["data"][120:6500])
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Relative amplitude')
        plt.show()

    def demodulate_and_integrate(self):
        #DOWN Conversion
        norm_factor = 1./(self.integration_length)
        input_vec_I = np.array(self.single_acq["single"]["acquisition"]["scope"]["path0"]["data"][self.start_sample:self.start_sample+self.integration_length])
        input_vec_Q = np.array(self.single_acq["single"]["acquisition"]["scope"]["path1"]["data"][self.start_sample:self.start_sample+self.integration_length])
        input_vec_I -= np.mean(input_vec_I)
        input_vec_Q -= np.mean(input_vec_Q)

        if self.mode == 'ssb':
            modulated_i = input_vec_I
            modulated_q = input_vec_Q
            time = np.arange(modulated_i.shape[0])*1e-9
            cosalpha = np.cos(2*np.pi*self.freq_if*time)
            sinalpha = np.sin(2*np.pi*self.freq_if*time)
            demod_matrix = 2*np.array([[cosalpha,-sinalpha],[sinalpha,cosalpha]])
            result = []
            for it,t,ii,qq in zip(np.arange(modulated_i.shape[0]),time,modulated_i,modulated_q):
                result.append(demod_matrix[:,:,it]@np.array([ii,qq]))
            demodulated_signal = np.array(result)
            integrated_signal = norm_factor*np.sum(demodulated_signal,axis=0)
            #print(integrated_signal,demodulated_signal[:,0].max()-demodulated_signal[:,0].min(),demodulated_signal[:,1].max()-demodulated_signal[:,1].min())
            self.integrated_signal = integrated_signal
            self.demodulated_signal = demodulated_signal.tolist()

        elif mode=='optimal':
            raise NotImplementedError('Optimal Demodulation Mode not coded yet.')
        else:
            raise NotImplementedError('Demodulation mode not understood.')
        return integrated_signal

	#Destructoras
    def _reset(self):
        #reset QRM
        self.qrm.reset()

    def stop(self):
	    #stop current sequence running in QRM
        self.qrm.stop_sequencer()

    def close(self):
	    #close connection to QRM
        self.qrm.close()


class Pulsar_QCM():

    def __init__(self, label, ip):
        """
        create Qblox QCM with name = label and connect to it in local IP = ip and set reference clock source
        Params format example:
                "ip": '192.168.0.3' (only 192.168.0.X accepted by Qblox)
                "label": "qcm"
        """
        self.label = label
        self.ip = ip
        self.qcm = pulsar_qcm(label, ip)

    #QCM Configuration method
    def setup(self, ref_clock):
        self._reset() #reset instrument from previous state
        self._set_reference_clock(ref_clock) #set reference clock source
        self.enable_sequencer_sync() #enable sync of QCM

    def _set_reference_clock(self, ref_clock):
        #set external reference clock to QCM
        self.qcm.reference_source(ref_clock)

    def enable_sequencer_sync(self):
        #enable sequencer sync
        self.qcm.sequencer0_sync_en(True)

    def set_waveforms(self, waveform_type, amplitude, IF_frequency, waveform_length, offset_i, offset_q, gain):
        #Setting QCM waveforms parameters
        self.wf_type = waveform_type
        self.amp = amplitude
        self.freq_if = IF_frequency
        self.leng = waveform_length
        self.offset_i = offset_i
        self.offset_q = offset_q
        self.gain = gain

    def modulate_envolope(self):
        #Waveform UP coversion
        std = self.leng/5
        print(f'parsed amp = {self.amp} V')
        if self.wf_type == 'Block':
            envelope_i = self.amp*np.ones(self.leng)
            envelope_q = self.amp*np.ones(self.leng)
        elif self.wf_type == 'Gaussian':
            envelope_i = self.amp*scipy.signal.gaussian(self.leng, std=std)
            envelope_q = np.zeros(self.leng) #amp*scipy.signal.gaussian(leng, std=std)
        time = np.arange(envelope_i.shape[0])*1e-9
        cosalpha = np.cos(2*np.pi*self.freq_if*time)
        sinalpha = np.sin(2*np.pi*self.freq_if*time)
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
        waveforms["modI_qcm_1"]["data"] = mod_signals[:,0]+self.offset_i
        waveforms["modQ_qcm_1"]["data"] = mod_signals[:,1]+self.offset_q
        self.waveforms = waveforms

    def set_sequence(self, seq):
        #set Q1ASM sequence to be executed in the QRM
        self.seq_prog = seq

    def upload_waveforms(self, acquisitions_qrm):
        # Reformat waveforms to lists if necessary.
        for name in self.waveforms:
            if str(type(self.waveforms[name]["data"]).__name__) == "ndarray":
                self.waveforms[name]["data"] = self.waveforms[name]["data"].tolist()  # JSON only supports lists

        #Add sequence program and waveforms to single dictionary and write to JSON file.
        wave_and_prog_dict = {"waveforms": self.waveforms, "weights":{}, "acquisitions": acquisitions_qrm, "program": self.seq_prog}
        with open("qcm_sequence.json", 'w', encoding='utf-8') as file:
            json.dump(wave_and_prog_dict, file, indent=4)
            file.close()
            self.qcm.sequencer0_waveforms_and_program(os.path.join(os.getcwd(), "qcm_sequence.json"))
            self.wave_and_prog_dict = wave_and_prog_dict

    def set_gain(self, gain):
        #set gain of the QCM
        #print(slef.qcm.sequencer0_gain_awg_path0())
        self.qcm.sequencer0_gain_awg_path0(gain)
        self.qcm.sequencer0_gain_awg_path1(gain)
        #print(qcm.pulsar_qcm.sequencer0_gain_awg_path0())

    def arm_and_start_sequencer(self):
        #arm sequencer and start playing sequence
        self.qcm.arm_sequencer()
        self.qcm.start_sequencer()
        self.status_start_seqencer0 = self.qcm.get_sequencer_state(0)

    def wait_sequencer_to_stop(self):
        #Wait for the sequencer to stop with a timeout period of one minute.
        self.pulsar_qcm.get_sequencer_state(0,1)


    #Destructoras
    def _reset(self):
         #reset QRM
        self.qcm.reset()

    def stop(self):
	    #stop current sequence running in QRM
        self.qcm.stop_sequencer()

    def close(self):
	    #close connection to QRM
        self.qcm.close()
