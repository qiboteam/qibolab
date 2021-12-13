import numpy as np
import json

from qibolab.instruments.rohde_schwarz import SGS100A
from qibolab.instruments.qblox import Pulsar_QCM
from qibolab.instruments.qblox import Pulsar_QRM

from qibolab.platforms.abstract import AbstractPlatform

class TIISingleQubit(AbstractPlatform):
    def load_default_settings(self):   
        # reorganise settings, all in one as a dictionary of devices with their names and settings
        self._general_settings = {
            'resonator_freq': 7.79813e9,
            'qubit_freq': 8.724e9,

            'data_folder': '.data/',
            "hardware_avg": 1024,
            "sampling_rate": 1e9,
            "software_averages": 1,
            "repetition_duration": 200000}  # Subject to calibration
        self._QRM_init_settings = {
            'ref_clock': 'external',                        # Clock source ['external', 'internal']
            'sequencer': 0,
            'sync_en': True,
            'hardware_avg_en': True,                        # If enabled, the device repeats the pulse (hardware_avg) times 
            'acq_trigger_mode': 'sequencer'}
        self._QRM_settings = {
            'gain': 0.35,   # Subject to calibration
            'hardware_avg': self._general_settings['hardware_avg'],
            'initial_delay': 0,
            "repetition_duration": 200000,
            'pulses': {
                'ro_pulse': {	"freq_if": 20e6,
                                "amplitude": 0.5,   # Subject to calibration
                                "start": 60+10, 
                                "length": 3000, # Subject to calibration
                                "offset_i": 0,
                                "offset_q": 0,
                                "shape": "Block",
                                "delay_before_readout": 4
                            }
                        },

            'start_sample': 100,    # Subject to calibration
            'integration_length': 2500,    # Subject to calibration
            'sampling_rate': self._general_settings['sampling_rate'],
            'mode': 'ssb'}
        self._QCM_init_settings = {
            'ref_clock': 'external',                        # Clock source ['external', 'internal']
            'sequencer': 0,
            'sync_en': True}
        self._QCM_settings = {
            'gain': 0.3,
            'hardware_avg': self._general_settings['hardware_avg'],
            'initial_delay': 0,
            "repetition_duration": 200000,
            'pulses': {
                'qc_pulse':{	"freq_if": 200e6,
                            "amplitude": 0.3,    # Subject to calibration
                            "start": 0,  
                            "length": 60,    # Subject to calibration
                            "offset_i": 0,
                            "offset_q": 0,
                            "shape": "Gaussian",
                            }
                        }}
        self._LO_QRM_settings = { 
            "power": 6,    # Subject to calibration
            "frequency":self._general_settings['resonator_freq'] - self._QRM_settings['pulses']['ro_pulse']['freq_if']}    # Subject to calibration
        self._LO_QCM_settings = { 
            "power": 6,    # Subject to calibration
            "frequency":self._general_settings['qubit_freq'] + self._QCM_settings['pulses']['qc_pulse']['freq_if']}    # Subject to calibration
        self._settings = {
            '_general_settings': self._general_settings,
            '_QRM_init_settings': self._QRM_init_settings,
            '_QRM_settings': self._QRM_settings,
            '_QCM_init_settings': self._QCM_init_settings,
            '_QCM_settings': self._QCM_settings,
            '_LO_QRM_settings': self._LO_QRM_settings,
            '_LO_QCM_settings': self._LO_QCM_settings,}

    def load_settings_from_file(self, file_name = 'tii_single_qubit_config.json'):
        #Read platform settings from json file
        config_file = open(file_name) 
        self._settings = json.load(config_file)
        self._general_settings = self._settings["_general_settings"]
        self._QRM_settings = self._settings["_QRM_settings"]
        self._QRM_init_settings = self._settings["_QRM_init_settings"]
        self._QCM_settings = self._settings["_QCM_settings"]
        self._QCM_init_settings = self._settings["_QCM_init_settings"]
        self._LO_QRM_settings = self._settings["_LO_QRM_settings"]
        self._LO_QCM_settings = self._settings["_LO_QCM_settings"]

    def save_settings_to_file(self):
        #save current platform settings to json file
       with open("tii_single_qubit_config.json", 'w', encoding='utf-8') as file:
            json.dump(self._settings, file, indent=4)
            file.close()

    def load_settings(self):
        # self.load_default_settings()
        self.load_settings_from_file()

    def __init__(self):
        self.load_settings()
        self._LO_qrm = SGS100A("LO_qrm", '192.168.0.7')
        self._LO_qcm = SGS100A("LO_qcm", '192.168.0.101')
        self._qrm = Pulsar_QRM("qrm", '192.168.0.2')
        self._qcm = Pulsar_QCM("qcm", '192.168.0.3')

    def setup(self):
        self._LO_qrm.setup(self._LO_QRM_settings)
        self._LO_qcm.setup(self._LO_QCM_settings)
        self._qrm.setup(self._QRM_settings)
        self._qcm.setup(self._QCM_settings)

    def stop(self):
        self._LO_qrm.off()
        self._LO_qcm.off()
        self._qrm.stop()
        self._qcm.stop()
        print("Closing")


    def __del__(self):
        self._LO_qrm.close()
        self._LO_qcm.close()
        self._qrm.close()
        self._qcm.close()
        print("Destructor Closing")

    def calibrate(self):
        # not implemented yet
        pass

    def get_calibration_settings(self):
        self.calibrate()
        return self._calibration_settings

    def execute(self, pulse_sequence, nshots):
        results = {}

        self.load_settings()
        self.setup()

        qrm = self._qrm
        qcm = self._qcm

        qrm.setup(qrm._settings)
        qrm.set_waveforms_from_pulses_definition(pulse_sequence.pulses[0])
        qrm.set_program_from_parameters(qrm._settings)
        qrm.set_acquisitions()
        qrm.set_weights()
        qrm.upload_sequence()
        
        qcm.setup(qcm._settings)
        qcm.set_waveforms_from_pulses_definition(pulse_sequence.pulses[1:])
        qcm.set_program_from_parameters(qcm._settings)
        qcm.set_acquisitions()
        qcm.set_weights()
        qcm.upload_sequence()
        
        self._LO_qrm.on()
        self._LO_qcm.on()
        qcm.play_sequence()
        acquisition_results = qrm.play_sequence_and_acquire()
        self.stop()

        return acquisition_results
