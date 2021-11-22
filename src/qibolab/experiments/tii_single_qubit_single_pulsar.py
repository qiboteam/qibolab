import numpy as np
import json


from tiiq.instruments.rohde_schwarz import SGS100A
from tiiq.instruments.qblox import Pulsar_QCM
from tiiq.instruments.qblox import Pulsar_QRM

# from rohde_schwarz import SGS100A
# from qblox import Pulsar_QCM
# from qblox import Pulsar_QRM

from quantify_core.data.handling import get_datadir, set_datadir
from quantify_core.measurement import MeasurementControl
from quantify_core.measurement.control import Settable, Gettable
from quantify_core.visualization.pyqt_plotmon import PlotMonitor_pyqt
from quantify_core.visualization.instrument_monitor import InstrumentMonitor

class TIISingleQubitSinglePulsar():

    _settings = {
        'data_dictionary': '.data/',
        "hardware_avg": 1024,
        "sampling_rate": 1e9,
        "software_averages": 1,
        "repetition_duration": 200000}
    _QRM_settings = {
            'gain': 0.4,
            'hardware_avg': _settings['hardware_avg'],
            'initial_delay': 0, 
            "repetition_duration": 200000,
            'pulses': {
                'qc_pulse':{	"freq_if": 200e6,
                                "amplitude": 0.2, 
                                "start": 0,        # cannot be <10?
                                "length": 4000,
                                "offset_i": 0,
                                "offset_q": 0,
                                "shape": "Gaussian",
                            },
                'ro_pulse': {	"freq_if": 20e6,
                                "amplitude": 0.9,
                                "start": 4000+40,  
                                "length": 3000,
                                "offset_i": 0,
                                "offset_q": 0,
                                "shape": "Block",
                            },
                        },
            'start_sample': 130,
            'integration_length': 2500,
            'sampling_rate': _settings['sampling_rate'],
            'mode': 'ssb'}
    _LO_QRM_settings = { "power": 15,
                        "frequency":7.79813e9 - _QRM_settings['pulses']['ro_pulse']['freq_if']}
    _LO_QCM_settings = { "power": 12,
                        "frequency":8.724e9 + _QRM_settings['pulses']['qc_pulse']['freq_if']}

    def __init__(self):

        #Read platform settings from json file
        #config = open('platform_config.json',) 
        #data = json.load(config)
        #_settings = data["_settings"]
        #_QRM_settings = data["_QRM_settings"]
        #_LO_QRM_settings = data["_LO_QRM_settings"]
        #_LO_QCM_settings = data["_LO_QCM_settings"]
    
        self._LO_qrm = SGS100A("LO_qrm", '192.168.0.7')
        self._LO_qcm = SGS100A("LO_qcm", '192.168.0.101')
        self._qrm = Pulsar_QRM("qrm", '192.168.0.2')

        self._MC = MeasurementControl('MC')
        self._plotmon = PlotMonitor_pyqt('Plot Monitor')
        self._insmon = InstrumentMonitor("Instruments Monitor")

        self._MC.instr_plotmon(self._plotmon.name)
        self._MC.instrument_monitor(self._insmon.name)

        set_datadir(self._settings['data_dictionary'])

    def setup(self):
        self._LO_qrm.setup(self._LO_QRM_settings)
        self._LO_qcm.setup(self._LO_QCM_settings)
        self._qrm.setup(self._QRM_settings)

    def stop(self):
        self._LO_qrm.off()
        self._LO_qcm.off()
        self._qrm.stop()

    def run_resonator_spectroscopy(self):
        self.setup()
        lowres_width = 30e6
        lowres_step = 2e6
        highres_width = 0.5e6
        highres_step = 0.05e6
        scanrange = np.concatenate((np.arange(-lowres_width,-highres_width,lowres_step),np.arange(-highres_width,highres_width,highres_step),np.arange(highres_width,lowres_width,lowres_step)))

        self._MC.settables(self._LO_qrm.LO.frequency)
        self._MC.setpoints(scanrange + self._LO_QRM_settings['frequency'])
        self._MC.gettables(Gettable(ROController(self._qrm)))
        self._LO_qrm.on()
        self._LO_qcm.off()
        dataset = self._MC.run('Resonator Spectroscopy', soft_avg = self._settings['software_averages'])
        self.stop()

    def run_qubit_spectroscopy(self):
        self.setup()
        lowres_width = 30e6
        lowres_step = 2e6
        highres_width = 5e6
        highres_step = 0.2e6
        scanrange = np.concatenate((np.arange(-lowres_width,-highres_width,lowres_step),np.arange(-highres_width,highres_width,highres_step),np.arange(highres_width,lowres_width,lowres_step)))

        self._MC.settables(self._LO_qcm.LO.frequency)
        self._MC.setpoints(scanrange + self._LO_QCM_settings['frequency'])
        self._MC.gettables(Gettable(ROController(self._qrm)))
        self._LO_qrm.on()
        self._LO_qcm.on()
        dataset = self._MC.run('Resonator Spectroscopy', soft_avg = self._settings['software_averages'])
        self.stop()

    def run_Rabi_pulse_length(self):
        self.setup()
        self._MC.settables(QCPulseLengthParameter(self._qrm))
        self._MC.setpoints(np.arange(1,3000,5))
        self._MC.gettables(Gettable(ROController(self._qrm)))
        self._LO_qrm.on()
        self._LO_qcm.on()
        dataset = self._MC.run('Rabi Pulse Length', soft_avg = self._settings['software_averages'])
        self.stop()

    def run_Rabi_pulse_gain(self):
        pass
    def run_t1(self):
        pass
    def run_ramsey(self):
        pass
    def run_spin_echo(self):
        pass

class ROController():

    # Quantify Gettable Interface Implementation
    label = ['Amplitude', 'Phase','I','Q']
    unit = ['V', 'Radians','V','V']
    name = ['A', 'Phi','I','Q']

    def __init__(self, qrm: Pulsar_QRM):
        self._qrm = qrm

    def get(self):
        qrm = self._qrm

        qrm.setup(qrm._settings)
        qrm.set_waveforms_from_pulses_definition(qrm._settings['pulses'])
        qrm.set_program_from_parameters(qrm._settings)
        qrm.set_acquisitions()
        qrm.set_weights()
        qrm.upload_sequence()

        return qrm.play_sequence_and_acquire()

class QCPulseLengthParameter():

    label = 'Qubit Control Pulse Length'
    unit = 'ns'
    name = 'qc_pulse_length'
    
    def __init__(self, qrm: Pulsar_QRM):
        self._qrm = qrm
        
    def set(self,value):
        self._qrm._settings['pulses']['qc_pulse']['length'] = value
        self._qrm._settings['pulses']['ro_pulse']['start'] = value+40
