import numpy as np

from tiiq.instruments.rohde_schwarz import SGS100A
from tiiq.instruments.qblox import Pulsar_QCM
from tiiq.instruments.qblox import Pulsar_QRM

from quantify_core.data.handling import get_datadir, set_datadir
from quantify_core.measurement import MeasurementControl
from quantify_core.measurement.control import Settable, Gettable
from quantify_core.visualization.pyqt_plotmon import PlotMonitor_pyqt
from quantify_core.visualization.instrument_monitor import InstrumentMonitor

class TIISingleQubit():

    _settings = {
        'data_dictionary': '.quantify-data/',
        "hardware_avg": 1024,
        "sampling_rate": 1e9,
        "software_averages": 5,
        "repetition_duration": 200000}
    _QRM_settings = {
            'gain': 0.5,
            'hardware_avg': _settings['hardware_avg'],
            'pulses': {
                'ro_pulse': {	"freq_if": 20e6,
                                "amplitude": 0.5, 
                                "length": 6000,
                                "offset_i": 0,
                                "offset_q": 0,
                                "shape": "Block",
                                "delay_before": 341,
                                "repetition_duration": _settings['repetition_duration'],
                            }
                        },

            'start_sample': 130,
            'integration_length': 600,
            'sampling_rate': _settings['sampling_rate'],
            'mode': 'ssb'}
    _QCM_settings = {
            'gain': 0.5,
            'hardware_avg': _settings['hardware_avg'],
            'pulses': {
                'qc_pulse':{	"freq_if": 100e6,
                            "amplitude": 0.25, 
                            "length": 300,
                            "offset_i": 0,
                            "offset_q": 0,
                            "shape": "Gaussian",
                            "delay_before": 1, # cannot be 0
                            "repetition_duration": _settings['repetition_duration'],
                            }
                        }}
    _LO_QRM_settings = { "power": 10,
                        "frequency":7.79813e9 - _QRM_settings['pulses']['ro_pulse']['freq_if']}
    _LO_QCM_settings = { "power": 12,
                        "frequency":8.72e9 + _QCM_settings['pulses']['qc_pulse']['freq_if']}

    def __init__(self):
        self._LO_qrm = SGS100A("LO_qrm", '192.168.0.7')
        self._LO_qcm = SGS100A("LO_qcm", '192.168.0.101')
        self._qrm = Pulsar_QRM("qrm", '192.168.0.2')
        self._qcm = Pulsar_QCM("qcm", '192.168.0.3')

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
        self._qcm.setup(self._QCM_settings)

    def stop(self):
        self._LO_qrm.off()
        self._LO_qcm.off()
        self._qrm.stop()
        self._qcm.stop()

    def run_resonator_spectroscopy(self):
        self.setup()
        lowres_width = 30e6
        lowres_step = 2e6
        highres_width = 0.5e6
        highres_step = 0.05e6
        scanrange = np.concatenate((np.arange(-lowres_width,-highres_width,lowres_step),np.arange(-highres_width,highres_width,highres_step),np.arange(highres_width,lowres_width,lowres_step)))

        self._MC.settables(self._LO_qrm.LO.frequency)
        self._MC.setpoints(scanrange + self._LO_QRM_settings['frequency'])
        self._MC.gettables(Gettable(ROAmplitudeController(self._qrm, self._qcm)))
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
        self._MC.gettables(Gettable(ROAmplitudeController(self._qrm, self._qcm)))
        self._LO_qrm.on()
        self._LO_qcm.on()
        dataset = self._MC.run('Resonator Spectroscopy', soft_avg = self._settings['software_averages'])
        self.stop()

    def run_Rabi_pulse_length(self):
        self.setup()
        self._MC.settables(PulseLengthParameter(self._qcm))
        self._MC.setpoints(np.arange(50,4000,10))
        self._MC.gettables(Gettable(ROAmplitudeController(self._qrm, self._qcm)))
        self._LO_qrm.on()
        self._LO_qcm.on()
        dataset = self._MC.run('Rabi Pulse Length', soft_avg = self._settings['software_averages'])
        self.stop()

class ROAmplitudeController():

    # Quantify Gettable Interface Implementation
    label = ['Amplitude', 'Phase','I','Q']
    unit = ['V', 'Radians','V','V']
    name = ['A', 'Phi','I','Q']

    def __init__(self, qrm: Pulsar_QRM, qcm: Pulsar_QCM):
        self._qrm = qrm
        self._qcm = qcm

    def get(self):
        qrm = self._qrm
        qcm = self._qcm

        qrm.setup(qrm._settings)
        qrm.set_waveforms_from_pulses_definition(qrm._settings['pulses'])
        qrm.set_program_from_parameters(qrm._settings)
        qrm.set_acquisitions()
        qrm.set_weights()
        qrm.upload_sequence()
        
        qcm.setup(qcm._settings)
        qcm.set_waveforms_from_pulses_definition(qcm._settings['pulses'])
        qcm.set_program_from_parameters(qcm._settings)
        qcm.set_acquisitions()
        qcm.set_weights()
        qcm.upload_sequence()
        
        # qcm.play_sequence() # if sync enabled I believe it is not necessary
        return qrm.play_sequence_and_acquire()

class PulseLengthParameter():

    label = 'Qubit Control Pulse Length'
    unit = 'ns'
    name = 'qc_pulse_length'
    
    def __init__(self, device):
        self._device = device
 #   def get(self):
 #       return settings['qc_pulse']['length']
        
    def set(self,value):
        self._device._settings['pulses']['qc_pulse']['length']=value