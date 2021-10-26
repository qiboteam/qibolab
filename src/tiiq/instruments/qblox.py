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

from qcodes import ManualParameter, Parameter

from pathlib import Path
from quantify.data.handling import get_datadir, set_datadir
from quantify.measurement import MeasurementControl
from quantify.measurement.control import Settable, Gettable
import quantify.visualization.pyqt_plotmon as pqm
from quantify.visualization.instrument_monitor import InstrumentMonitor

from pulsar_qcm.pulsar_qcm import pulsar_qcm
from pulsar_qrm.pulsar_qrm import pulsar_qrm


class Pulsar_QRM():

	# Construction method
    def __init__(self, label, ip, ref_clock):
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
    def setup(QRM_info: dict):
        '''
        Function for setting up the Qblox QRM parameters
        Input Params: amplitude, IF freq, length, offset I, offset Q
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

        self._reset() #reset instrument from previous state
        self._set_reference_clock(QRM_info[ref_clock]) #set reference clock source
        self.set_data_dictionary(QRM_info[data_dictionary]) #set data directory for generated waveforms

        #Set up instrument integration and modulation parameters
        self.start_sample = QRM_info[start_sample]
        self.hardware_avg = QRM_info[hardware_avg]
        self.integration_length = QRM_info[integration_length]
        self.sampling_rate = QRM_info[sampling_rate]
        self.mode = QRM_info[mode]

    #Modifiers
    def set_data_dictionary(self, data_dict):
        set_datadir(data_dict)
        print(f"Data will be saved in:\n{get_datadir()}")
        return get_datadir()

    def _set_reference_clock(self, ref_clock):
        #set external reference clock QRM and QCM
        self.qrm.reference_source(ref_clock)

    def set_waveforms(self, IF_frequency, waveform_length, offset_i, offset_q, amplitude):
        #Setting waveform parameters
        self.freq_if = IF_frequency
        self.leng = waveform_length
        self.offset_i = offset_i
        self.offset_q = offset_q
        self.amp = amplitude

    def modulate_envolope(self):
        #Waveform UP coversion
        envelope_i = amp*np.ones(self.leng)
        envelope_q = amp*np.zeros(self.leng)
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

    def specify_acquisitions(self):
        #define type Qblox QRM acquisitions. See Qblox documentation to understand format
        acquisitions = {"single":   {"num_bins": 1, "index":0}}
        self.acquisitions = acquisitions

    def set_sequence(self, seq):
        #set Q1ASM sequence to be executed in the QRM
        self.seq_prog = seq

    def upload_waveforms(self):
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

    def enable_hardware_averaging(self):
        #Enable QRM acquisition average mode
        self.qrm.scope_acq_sequencer_select(0)
        self.qrm.scope_acq_avg_mode_en_path0(True)
        self.qrm.scope_acq_avg_mode_en_path1(True)

    def configure_sequencer_sync(self):
        self.qrm.sequencer0_sync_en(True)

	#Destructoras
	def _reset (self):
        #reset QRM
        self.pulsar_qrm.reset()

	def stop(self):
	    #stop current sequence running in QRM
        self.pulsar_qrm.stop_sequencers()

	def close(self):
	    #close connection to QRM
        self.pulsar_qrm.close()


class Pulsar_QCM():
    #_ip_address = None
    #_qrm = None

    #_clock_type = "Internal"
    #_gain = None
    #_offset = None
    #_phase = None
    #_sequencers = []
    #_paths = []
    #_ports = []
    #Out1 = None
    #Out2 = None
    #In1 = None
    #In2 = None

	# Contructora
    def __init__(self, label, ip):
        """
        create Qblox QCM (qblox read out module) with name = label and connect to it in local IP = ip
        Params format example:
                "ip": '192.168.0.2' (only 192.168.0.X accepted by Qblox)
                "label": "qrm"
        """
        self.label = label
        self.ip = ip
        self.qcm = pulsar_qrm(label, ip)

	#Modificadoras

    def set_reference_clock_external (self):
    #set external reference clock QRM and QCM

	def setup (self, params):
	#Function for setting up the QRM waveforms

	def upload_waveforms(self):
	#Function for upload waveforms in QRM

	def upload_sequence(self, type, params):
	#Upload the sequence to the QRM necessary for experiment defined in type (Raby, T1, T2...)

	#Destructoras

	def reset (self):
	#reset QRM

	def stop_sequencers(self):
	#stop current sequence running in QRM

	def close_connections(self):
	#close connection to QRM
