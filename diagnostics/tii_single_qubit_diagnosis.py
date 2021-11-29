import numpy as np
from scipy.signal import savgol_filter
import qibolab.calibration.fitting 

import matplotlib.pyplot as plt

import xarray as xr

from quantify_core.data.handling import get_datadir, set_datadir
from quantify_core.measurement import MeasurementControl
from quantify_core.measurement.control import Settable, Gettable
from quantify_core.visualization.pyqt_plotmon import PlotMonitor_pyqt
from quantify_core.visualization.instrument_monitor import InstrumentMonitor

from qcodes import ManualParameter, Parameter
from qcodes.instrument import Instrument

from qibolab.instruments.qblox import Pulsar_QCM
from qibolab.instruments.qblox import Pulsar_QRM

from qibolab.platforms.tii_single_qubit import TIISingleQubit

tiisq = TIISingleQubit()

MC = MeasurementControl('MC')
plotmon = PlotMonitor_pyqt('Plot Monitor')
plotmon.tuids_max_num(3)
insmon = InstrumentMonitor("Instruments Monitor")
MC.instr_plotmon(plotmon.name)
MC.instrument_monitor(insmon.name)
set_datadir('.data/')

def variable_resolution_scanrange(lowres_width, lowres_step, highres_width, highres_step):
    #[.     .     .     .     .     .][...................]0[...................][.     .     .     .     .     .]
    #[-------- lowres_width ---------][-- highres_width --] [-- highres_width --][-------- lowres_width ---------]
    #>.     .< lowres_step
    #                                 >..< highres_step
    #                                                      ^ centre value = 0
    scanrange = np.concatenate(
        (   np.arange(-lowres_width,-highres_width,lowres_step),
            np.arange(-highres_width,highres_width,highres_step),
            np.arange(highres_width,lowres_width,lowres_step)
        )
    )
    return scanrange

def run_resonator_spectroscopy():
    tiisq.load_default_settings()
    tiisq.setup()
    
    # Fast Sweep
    tiisq._settings['software_averages'] = 1
    scanrange = variable_resolution_scanrange(lowres_width= 30e6, lowres_step= 1e6, highres_width= 1e6, highres_step= 0.1e6)

    MC.settables(tiisq._LO_qrm.LO.frequency)
    MC.setpoints(scanrange + tiisq._LO_QRM_settings['frequency'])
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.off()
    dataset = MC.run('Resonator Spectroscopy Fast', soft_avg = tiisq._settings['software_averages'])
    # http://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html
    tiisq.stop()
    tiisq._LO_QRM_settings['frequency'] = dataset['x0'].values[dataset['y0'].argmax().values]
    
    # Precision Sweep
    tiisq._settings['software_averages'] = 1 # 3
    scanrange = np.arange(-0.5e6, 0.5e6, 0.02e6)
    MC.settables(tiisq._LO_qrm.LO.frequency)
    MC.setpoints(scanrange + tiisq._LO_QRM_settings['frequency'])
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.off()
    dataset = MC.run('Resonator Spectroscopy Precision', soft_avg = tiisq._settings['software_averages'])
    tiisq.stop()

    smooth_dataset = savgol_filter(dataset['y0'].values, 25, 2)
    tiisq._LO_QRM_settings['frequency'] = dataset['x0'].values[smooth_dataset.argmax()]
    tiisq.resonator_freq = dataset['x0'].values[smooth_dataset.argmax()] + tiisq._QRM_settings['pulses']['ro_pulse']['freq_if']
    print(f"Resonator Frequency = {tiisq.resonator_freq}")

    print(len(dataset['y0'].values))
    print(len(smooth_dataset))

    fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
    ax.plot(dataset['x0'].values, dataset['y0'].values,'-',color='C0')
    ax.plot(dataset['x0'].values, smooth_dataset,'-',color='C1')
    ax.title.set_text('Original')
    #ax.xlabel("Frequency")
    #ax.ylabel("Amplitude")
    ax.plot(dataset['x0'].values[smooth_dataset.argmax()], smooth_dataset[smooth_dataset.argmax()], 'o', color='C2')

    # determine off-resonance amplitude and typical noise

    return dataset

def run_qubit_spectroscopy():
    tiisq.load_default_settings()
    tiisq._QCM_settings['pulses']['qc_pulse']['length'] = 4000
    tiisq._QRM_settings['pulses']['ro_pulse']['start'] = tiisq._QCM_settings['pulses']['qc_pulse']['length']+40
    tiisq.setup()

    # Fast Sweep
    #scanrange = variable_resolution_scanrange(lowres_width= 30e6, lowres_step= 2e6, highres_width= 2e6, highres_step= 0.2e6)
    scanrange = np.arange(-300e6, 100e6, 1e6)

    MC.settables(tiisq._LO_qcm.LO.frequency)
    MC.setpoints(scanrange + tiisq._LO_QCM_settings['frequency'])
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.on()
    dataset = MC.run('Qubit Spectroscopy Fast', soft_avg = tiisq._settings['software_averages'])
    tiisq.stop()

    tiisq._LO_QCM_settings['frequency'] = dataset['x0'].values[dataset['y0'].argmin().values]
    

    # Precision Sweep
    tiisq._settings['software_averages'] = 3
    scanrange = np.arange(-30e6, 30e6, 0.5e6)
    MC.settables(tiisq._LO_qcm.LO.frequency)
    MC.setpoints(scanrange + tiisq._LO_QCM_settings['frequency'])
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.on()
    dataset = MC.run('Qubit Spectroscopy Precision', soft_avg = tiisq._settings['software_averages'])
    tiisq.stop()

    smooth_dataset = savgol_filter(dataset['y0'].values, 11, 2)
    tiisq._LO_QCM_settings['frequency'] = dataset['x0'].values[smooth_dataset.argmin()]
    print(dataset['x0'].values[smooth_dataset.argmin()])
    tiisq.qubit_freq = dataset['x0'].values[smooth_dataset.argmin()] - tiisq._QCM_settings['pulses']['qc_pulse']['freq_if']
    print(f"Qubit Frequency = {tiisq.qubit_freq}")

    print(len(dataset['y0'].values))
    print(len(smooth_dataset))

    fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
    ax.plot(dataset['x0'].values, dataset['y0'].values,'-',color='C0')
    ax.plot(dataset['x0'].values, smooth_dataset,'-',color='C1')
    ax.title.set_text('Original')
    #ax.xlabel("Frequency")
    #ax.ylabel("Amplitude")
    ax.plot(dataset['x0'].values[smooth_dataset.argmin()], smooth_dataset[smooth_dataset.argmin()], 'o', color='C2')

    return dataset

def run_Rabi_pulse_length():
    tiisq.load_default_settings()
    tiisq._LO_QRM_settings['frequency'] = tiisq.resonator_freq - tiisq._QRM_settings['pulses']['ro_pulse']['freq_if']
    tiisq._LO_QCM_settings['frequency'] = tiisq.qubit_freq + tiisq._QCM_settings['pulses']['qc_pulse']['freq_if']
    tiisq.setup()
    MC.settables(QCPulseLengthParameter(tiisq._qrm, tiisq._qcm))
    MC.setpoints(np.arange(1,2000,5))
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.on()
    dataset = MC.run('Rabi Pulse Length', soft_avg = tiisq._settings['software_averages'])
    tiisq.stop()

def run_Rabi_pulse_gain():
    tiisq.load_default_settings()
    tiisq._LO_QRM_settings['frequency'] = tiisq.resonator_freq - tiisq._QRM_settings['pulses']['ro_pulse']['freq_if']
    tiisq._LO_QCM_settings['frequency'] = tiisq.qubit_freq + tiisq._QCM_settings['pulses']['qc_pulse']['freq_if']
    tiisq.setup()
    MC.settables(QCPulseGainParameter(tiisq._qcm))
    MC.setpoints(np.arange(0,1,0.02))
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.on()
    dataset = MC.run('Rabi Pulse Gain', soft_avg = tiisq._settings['software_averages'])
    tiisq.stop()

def run_Rabi_pulse_length_and_gain():
    tiisq.load_default_settings()
    tiisq._LO_QRM_settings['frequency'] = tiisq.resonator_freq - tiisq._QRM_settings['pulses']['ro_pulse']['freq_if']
    tiisq._LO_QCM_settings['frequency'] = tiisq.qubit_freq + tiisq._QCM_settings['pulses']['qc_pulse']['freq_if']
    tiisq.setup()
    MC.settables([QCPulseLengthParameter(tiisq._qrm, tiisq._qcm), QCPulseGainParameter(tiisq._qcm)])
    setpoints_length = np.arange(1,200,10)
    setpoints_gain = np.arange(0,100,5)
    MC.setpoints_grid([setpoints_length,setpoints_gain])
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.on()
    dataset = MC.run('Rabi Pulse Length and Gain', soft_avg = tiisq._settings['software_averages'])
    # Analyse data to look for the smallest qc_pulse length that renders off-resonance amplitude, determine corresponding pi_pulse gain
    # platform.pi_pulse_length = 
    # platform.pi_pulse_gain = 

def run_Rabi_pulse_length_and_amplitude():
    tiisq.load_default_settings()
    tiisq._LO_QRM_settings['frequency'] = tiisq.resonator_freq - tiisq._QRM_settings['pulses']['ro_pulse']['freq_if']
    tiisq._LO_QCM_settings['frequency'] = tiisq.qubit_freq + tiisq._QCM_settings['pulses']['qc_pulse']['freq_if']
    tiisq.setup()
    MC.settables([QCPulseLengthParameter(tiisq._qrm, tiisq._qcm), QCPulseAmplitudeParameter(tiisq._qcm)])
    setpoints_length = np.arange(1,200,10)
    setpoints_amplitude = np.arange(0,100,5)
    MC.setpoints_grid([setpoints_length,setpoints_amplitude])
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.on()
    dataset = MC.run('Rabi Pulse Length and Gain', soft_avg = tiisq._settings['software_averages'])
    # Analyse data to look for the smallest qc_pulse length that renders off-resonance amplitude, determine corresponding pi_pulse gain
    # platform.pi_pulse_length = 
    # platform.pi_pulse_gain = 

def run_t1():
    tiisq.load_default_settings()
    tiisq._LO_QRM_settings['frequency'] = tiisq.resonator_freq - tiisq._QRM_settings['pulses']['ro_pulse']['freq_if']
    tiisq._LO_QCM_settings['frequency'] = tiisq.qubit_freq + tiisq._QCM_settings['pulses']['qc_pulse']['freq_if']
    tiisq._QCM_settings['pulses']['qc_pulse']['length'] = tiisq.pi_pulse_length
    tiisq._QCM_settings['gain'] = tiisq.pi_pulse_gain
    tiisq.setup()
    MC.settables(T1WaitParameter(tiisq._qrm))
    MC.setpoints(np.arange(4,500,10))
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.on()
    dataset = MC.run('T1', soft_avg = tiisq._settings['software_averages'])
    tiisq.stop()
    # fit data and determine T1
    # platform.t1 = 

def run_ramsey():
    tiisq.load_default_settings()
    tiisq._LO_QRM_settings['frequency'] = tiisq.resonator_freq - tiisq._QRM_settings['pulses']['ro_pulse']['freq_if']
    tiisq._LO_QCM_settings['frequency'] = tiisq.qubit_freq + tiisq._QCM_settings['pulses']['qc_pulse']['freq_if']
    tiisq._QCM_settings['gain'] = tiisq.pi_pulse_gain
    tiisq._QCM_settings['pulses'] = {
            'qc_pulse':{	"freq_if": 200e6,
                        "amplitude": 0.3,
                        "start": 0,  
                        "length": tiisq.pi_pulse_length//2,    
                        "offset_i": 0,
                        "offset_q": 0,
                        "shape": "Gaussian",
                        },
            'qc2_pulse':{	"freq_if": 200e6,
                        "amplitude": 0.3,    
                        "start": tiisq.pi_pulse_length//2 + 0,  
                        "length": tiisq.pi_pulse_length//2,
                        "offset_i": 0,
                        "offset_q": 0,
                        "shape": "Gaussian",
                        }
    }
    tiisq.setup()
    MC.settables(RamseyWaitParameter(tiisq._qrm, tiisq._qcm))
    MC.setpoints(np.arange(4,500,10))
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.on()
    dataset = MC.run('Ramsey', soft_avg = tiisq._settings['software_averages'])
    tiisq.stop()
    # fit data and determine Ramsey Time and dephasing
    # platform.ramsey = 
    # platform.qubit_freq += dephasing

def run_spin_echo():
    tiisq.load_default_settings()
    tiisq._LO_QRM_settings['frequency'] = tiisq.resonator_freq - tiisq._QRM_settings['pulses']['ro_pulse']['freq_if']
    tiisq._LO_QCM_settings['frequency'] = tiisq.qubit_freq + tiisq._QCM_settings['pulses']['qc_pulse']['freq_if']
    tiisq._QCM_settings['gain'] = tiisq.pi_pulse_gain
    tiisq._QCM_settings['pulses'] = {
            'qc_pulse':{	"freq_if": 200e6,
                        "amplitude": 0.3,
                        "start": 0,  
                        "length": tiisq.pi_pulse_length//2,    
                        "offset_i": 0,
                        "offset_q": 0,
                        "shape": "Gaussian",
                        },
            'qc2_pulse':{	"freq_if": 200e6,
                        "amplitude": 0.3,    
                        "start": tiisq.pi_pulse_length//2 + 0,  
                        "length": tiisq.pi_pulse_length,
                        "offset_i": 0,
                        "offset_q": 0,
                        "shape": "Gaussian",
                        }
    }
    tiisq.setup()
    MC.settables(SpinEchoWaitParameter(tiisq._qrm, tiisq._qcm))
    MC.setpoints(np.arange(4,500,10))
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.on()
    dataset = MC.run('Spin Echo', soft_avg = tiisq._settings['software_averages'])
    tiisq.stop()
    # ?

# run resonator shifted spectroscopy with pi pulse

class ROController():

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
        
        qcm.play_sequence()
        acquisition_results = qrm.play_sequence_and_acquire()
        return acquisition_results

class QCPulseLengthParameter():

    label = 'Qubit Control Pulse Length'
    unit = 'ns'
    name = 'qc_pulse_length'
    
    def __init__(self, qrm: Pulsar_QRM, qcm: Pulsar_QCM):
        self._qrm = qrm
        self._qcm = qcm
        
    def set(self,value):
        self._qcm._settings['pulses']['qc_pulse']['length']=value
        self._qrm._settings['pulses']['ro_pulse']['start']=value+4

class QCPulseGainParameter():

    label = 'Qubit Control Gain'
    unit = '%'
    name = 'qc_pulse_gain'
    
    def __init__(self, qcm: Pulsar_QCM):
        self._qcm = qcm
        
    def set(self,value):
        sequencer = self._qcm._settings['sequencer']
        gain = value / 100
        if sequencer == 1:
            self._qcm._qcm.sequencer1_gain_awg_path0(gain)
            self._qcm._qcm.sequencer1_gain_awg_path1(gain)
        else:
            self._qcm._qcm.sequencer0_gain_awg_path0(gain)
            self._qcm._qcm.sequencer0_gain_awg_path1(gain)

class QCPulseAmplitudeParameter():

    label = 'Qubit Control Pulse Amplitude'
    unit = '%'
    name = 'qc_pulse_amplitude'
    
    def __init__(self, qcm: Pulsar_QCM):
        self._qcm = qcm
        
    def set(self,value):
        self._qcm._settings['pulses']['qc_pulse']['amplitude']=value /100

class T1WaitParameter():
    label = 'Time'
    unit = 'ns'
    name = 't1_wait'
    initial_value = 0
    
    def __init__(self, qrm: Pulsar_QRM):
        self._qrm = qrm
        
    def set(self,value):
        #must be >= 4ns <= 65535
        self._qrm._settings['pulses']['ro_pulse']['delay_before_readout'] = value
    
class RamseyWaitParameter():
    label = 'Time'
    unit = 'ns'
    name = 'ramsey_wait'
    initial_value = 0
    
    def __init__(self, qrm: Pulsar_QRM, qcm: Pulsar_QCM):
        self._qrm = qrm
        self._qcm = qcm
        
    def set(self,value):
        self._qcm._settings['pulses']['qc2_pulse']['start'] = tiisq.pi_pulse_length//2 + value
        self._qrm._settings['pulses']['ro_pulse']['start']= tiisq.pi_pulse_length + value + 4
        
class SpinEchoWaitParameter():
    label = 'Time'
    unit = 'ns'
    name = 'spin_echo_wait'
    initial_value = 0
    
    def __init__(self, qrm: Pulsar_QRM, qcm: Pulsar_QCM):
        self._qrm = qrm
        self._qcm = qcm
        
    def set(self,value):
        self._qcm._settings['pulses']['qc2_pulse']['start'] = tiisq.pi_pulse_length//2 + value
        self._qrm._settings['pulses']['ro_pulse']['start']= 3 * tiisq.pi_pulse_length//2 + 2 * value + 4

# T1: RX(pi) - wait t(rotates z) - readout
# Ramsey: RX(pi/2) - wait t(rotates z) - RX(pi/2) - readout
# Spin Echo: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - readout
# Spin Echo: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - RX(pi/2) - readout

# Ignore all functions after this point
# Check Ramiros code to generate sequences
    #def sequence_program_single(self):
    #def sequence_program_average(self):
    #def sequence_program_qubit_spec(self,qcm_leng,repetition_duration=200000):
    #def sequence_program_t1(self,qcm_leng,wait_time_ns=20,repetition_duration=200000):
    #def sequence_program_echo(self,qcm_leng,wait_time_ns=20,repetition_duration=200000):
    #def sequence_program_ramsey(self,qcm_leng,wait_time_ns=20,repetition_duration=200000):
