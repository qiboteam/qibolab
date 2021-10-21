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

        #No se si deberia ir aqui esto????? set up instrument integration and modulation
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


	#Destructoras
	def _reset (self):
	#reset QRM
    self.pulsar_qrm.reset()

	def stop(self):
	#stop current sequence running in QRM
        self.pulsar_qrm.stop_sequencer()

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
