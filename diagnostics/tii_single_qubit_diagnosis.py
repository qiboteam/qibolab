import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
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

import pyqtgraph.multiprocess as pgmp
from qcodes.plots.pyqtgraph import QtPlot, TransformState
import matplotlib.colors as mplc

from qibolab.instruments.qblox import Pulsar_QCM
from qibolab.instruments.qblox import Pulsar_QRM
from qibolab.instruments.rohde_schwarz import SGS100A

from qibolab.platforms.tii_single_qubit import TIISingleQubit

tiisq = TIISingleQubit()

MC = MeasurementControl('MC')
plotmon = PlotMonitor_pyqt('Plot Monitor')
plotmon.tuids_max_num(3)
insmon = InstrumentMonitor("Instruments Monitor")
MC.instr_plotmon(plotmon.name)
MC.instrument_monitor(insmon.name)
set_datadir('.data/')


## TODOs ##
# Change settable clases so that sweeps are done without changing settings
# Pass the turning on of the LO_qrm and LO_qcm to the ROController class
# Implement pulses class
# Review and rewrite wave and program generation classes
#   - pulses in parallel
# Implement a script to optimise readout paraments with live plotting of incoming pulses
# Implement run_shifted_resonator_spectroscopy to see the cavity resonance when the qubit is excited with a pi pulse
# Pulse phases

def test_plot(dset: xr.Dataset):
    pw = PlotWindow(window_title = "Test Plot")
    pw.add_trace(
        tuid = 'pulse envelope',
        x=dset['x0'].values,
        y=dset['y0'].values,
        subplot=1,
        xlabel=dset['x0'].attrs["long_name"],
        xunit=dset['x0'].attrs["units"],
        ylabel=dset['y0'].attrs["long_name"],
        yunit=dset['y0'].attrs["units"],
        symbol="o",
        symbolSize=6,
        color=mplc.to_rgb("#1f77b4"),
        name= 'pulse envelope'
    )
    pw.update(tuid = 'pulse envelope', x = dset['x0'].values, y = dset['y0'].values)

class PlotWindow():
    def __init__(self, window_title):
        self.proc = pgmp.QtProcess(processRequests=False)
        timeout = 60
        self._remote_ppr = self.proc._import("tii_single_qubit_diagnosis", timeout=timeout)
        self._remote_plot = self._remote_ppr.PlotWindowRemote(window_title = window_title)
    def add_trace(self, tuid, x, y, subplot, xlabel, xunit, ylabel, yunit, symbol, symbolSize, color, name):
        self._remote_plot.add_trace(
                        tuid = tuid,
                        x=x,
                        y=y,
                        subplot=subplot,
                        xlabel=xlabel,
                        xunit=xunit,
                        ylabel=ylabel,
                        yunit=yunit,
                        symbol=symbol,
                        symbolSize=symbolSize,
                        color=color,
                        name=name
                    )
    def clear(self):
        self._remote_plot.clear()
    def update(self, tuid, x, y):
        self._remote_plot.update(tuid, x, y)

class PlotWindowRemote():
    curves = {}
    def __init__(self, window_title = "Plot"):
        self.main_QtPlot = QtPlot(
            window_title=window_title,
            figsize=(600, 400),
            remote=False,
        )
    def add_trace(self, tuid, x, y, subplot, xlabel, xunit, ylabel, yunit, symbol, symbolSize, color, name):
        self.main_QtPlot.add(
                        x=x,
                        y=y,
                        subplot=subplot,
                        xlabel=xlabel,
                        xunit=xunit,
                        ylabel=ylabel,
                        yunit=yunit,
                        symbol=symbol,
                        symbolSize=symbolSize,
                        color=color,
                        name=name
                    )
        self.curves[tuid] = self.main_QtPlot.traces[-1]

    def clear(self):
        if self.main_QtPlot.traces:
            self.main_QtPlot.clear()
    def update(self, tuid, x, y):
        self.curves[tuid]["config"]["x"] = x
        self.curves[tuid]["config"]["y"] = y
        self.main_QtPlot.update_plot()

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
    tiisq.load_settings()
    tiisq.setup()
    
    # Fast Sweep
    tiisq._general_settings['software_averages'] = 1 
    scanrange = variable_resolution_scanrange(lowres_width= 30e6, lowres_step= 2e6, highres_width= 1e6, highres_step= 0.2e6)

    MC.settables(tiisq._LO_qrm.LO.frequency)
    MC.setpoints(scanrange + tiisq._LO_QRM_settings['frequency'])
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.off()
    dataset = MC.run('Resonator Spectroscopy Fast', soft_avg = tiisq._general_settings['software_averages'])
    # http://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html
    tiisq.stop()
    tiisq._LO_QRM_settings['frequency'] = dataset['x0'].values[dataset['y0'].argmax().values]
    tiisq._general_settings['resonator_spectroscopy_avg_min_ro_voltage'] = np.mean(dataset['y0'].values[:25]) * 1e6
    
    # Precision Sweep
    tiisq._general_settings['software_averages'] = 1 # 3
    scanrange = np.arange(-0.5e6, 0.5e6, 0.02e6)
    MC.settables(tiisq._LO_qrm.LO.frequency)
    MC.setpoints(scanrange + tiisq._LO_QRM_settings['frequency'])
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.off()
    dataset = MC.run('Resonator Spectroscopy Precision', soft_avg = tiisq._general_settings['software_averages'])
    tiisq.stop()

    smooth_dataset = savgol_filter(dataset['y0'].values, 25, 2)
    tiisq._LO_QRM_settings['frequency'] = dataset['x0'].values[smooth_dataset.argmax()]
    tiisq._general_settings['resonator_freq'] = dataset['x0'].values[smooth_dataset.argmax()] + tiisq._QRM_settings['pulses']['ro_pulse']['freq_if']
    # tiisq._general_settings['max_ro_voltage'] = dataset['y0'].max().item() * 1e6
    tiisq._general_settings['resonator_spectroscopy_max_ro_voltage'] = smooth_dataset.max() * 1e6
    print('\n')
    print(f"Resonator Frequency = {tiisq._general_settings['resonator_freq']}")
    print(f"Maximum Voltage Measured = { tiisq._general_settings['resonator_spectroscopy_max_ro_voltage']} μV")
    print(f"Average Minimum Voltage Measured = {tiisq._general_settings['resonator_spectroscopy_avg_min_ro_voltage']} μV")

    tiisq.save_settings_to_file() # instead of saving all parameters, it would be safer to save _general_settings['resonator_freq'] & _LO_QRM_settings['frequency']


    # print(len(dataset['y0'].values))
    # print(len(smooth_dataset))

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
    tiisq.load_settings()
    tiisq._QCM_settings['pulses']['qc_pulse']['length'] = 4000
    tiisq._QRM_settings['pulses']['ro_pulse']['start'] = tiisq._QCM_settings['pulses']['qc_pulse']['length']+4
    tiisq.setup()
    
    
    # Fast Sweep
    #scanrange = variable_resolution_scanrange(lowres_width= 30e6, lowres_step= 2e6, highres_width= 2e6, highres_step= 0.2e6)
    tiisq._general_settings['software_averages'] = 1
    scanrange = np.arange(-400e6, 100e6, 2e6)

    MC.settables(tiisq._LO_qcm.LO.frequency)
    MC.setpoints(scanrange + tiisq._LO_QCM_settings['frequency'])
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.on()
    dataset = MC.run('Qubit Spectroscopy Fast', soft_avg = tiisq._general_settings['software_averages'])
    tiisq.stop()

    # tiisq._LO_QCM_settings['frequency'] = dataset['x0'].values[dataset['y0'].argmin().values]
    

    # Precision Sweep
    tiisq._general_settings['software_averages'] = 3
    scanrange = np.arange(-30e6, 30e6, 0.5e6)
    MC.settables(tiisq._LO_qcm.LO.frequency)
    MC.setpoints(scanrange + tiisq._LO_QCM_settings['frequency'])
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.on()
    dataset = MC.run('Qubit Spectroscopy Precision', soft_avg = tiisq._general_settings['software_averages'])
    tiisq.stop()

    smooth_dataset = savgol_filter(dataset['y0'].values, 11, 2)
    tiisq._LO_QCM_settings['frequency'] = dataset['x0'].values[smooth_dataset.argmin()]
    print(dataset['x0'].values[smooth_dataset.argmin()])
    tiisq._general_settings['qubit_freq'] = dataset['x0'].values[smooth_dataset.argmin()] - tiisq._QCM_settings['pulses']['qc_pulse']['freq_if']
    tiisq._general_settings['qubit_spectroscopy_min_ro_voltage'] = smooth_dataset.min() * 1e6
    print('\n')
    print(f"Qubit Frequency = {tiisq._general_settings['qubit_freq']}")
    print(f"Minimum Voltage Measured = {tiisq._general_settings['qubit_spectroscopy_min_ro_voltage']} μV")

    tiisq.save_settings_to_file()

    # print(len(dataset['y0'].values))
    # print(len(smooth_dataset))

    fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
    ax.plot(dataset['x0'].values, dataset['y0'].values,'-',color='C0')
    ax.plot(dataset['x0'].values, smooth_dataset,'-',color='C1')
    ax.title.set_text('Original')
    #ax.xlabel("Frequency")
    #ax.ylabel("Amplitude")
    ax.plot(dataset['x0'].values[smooth_dataset.argmin()], smooth_dataset[smooth_dataset.argmin()], 'o', color='C2')

    return dataset

def run_Rabi_pulse_length():
    tiisq.load_settings()
    tiisq.setup()
    tiisq._general_settings['software_averages'] = 3
    MC.settables(QCPulseLengthParameter(tiisq._qrm, tiisq._qcm))
    MC.setpoints(np.arange(1,200,1))
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.on()
    dataset = MC.run('Rabi Pulse Length', soft_avg = tiisq._general_settings['software_averages'])
    tiisq.stop()
    pi_pulse_duration = fit_rabi(dataset['y0'].values, dataset['x0'].values)
    print(pi_pulse_duration)




def rabi(x, *p) :
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   Period    T                  : p[2]
    #   Phase                        : p[3]
    #   Arbitrary parameter T_2      : p[4]
    return p[0] + p[1] * np.sin(2 * np.pi / p[2] * x + p[3]) * np.exp(-x / p[4])

def sin(x, *p) :
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   Period    T                  : p[2]
    #   Phase                        : p[3]
    return p[0] + p[1] * np.sin(2 * np.pi / p[2] * x + p[3])

def fit_rabi(amp_array, time_array):
    pguess = [
        np.mean(amp_array),
        max(amp_array) - min(amp_array),
        35e-9,
        np.pi/2,
        0.1e-6
    ]
    popt, pcov = curve_fit(rabi, time_array, amp_array, p0=pguess)
    pi_pulse_duration = popt[2] / 2
    return (pi_pulse_duration, popt, pcov)




def run_Rabi_pulse_gain():
    tiisq.load_settings()
    tiisq._QCM_settings['pulses']['qc_pulse']['length'] = 200
    tiisq._QRM_settings['pulses']['ro_pulse']['start'] = tiisq._QCM_settings['pulses']['qc_pulse']['length']+4
    tiisq._LO_QCM_settings['power'] = 10
    tiisq.setup()
    MC.settables(QCPulseGainParameter(tiisq._qcm))
    MC.setpoints(np.arange(0,50,1))
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.on()
    dataset = MC.run('Rabi Pulse Gain', soft_avg = tiisq._general_settings['software_averages'])
    tiisq.stop()

def run_Rabi_pulse_length_and_gain():
    tiisq.load_settings()
    tiisq._LO_QRM_settings['frequency'] = tiisq._general_settings['resonator_freq'] - tiisq._QRM_settings['pulses']['ro_pulse']['freq_if']
    tiisq._LO_QCM_settings['frequency'] = tiisq._general_settings['qubit_freq'] + tiisq._QCM_settings['pulses']['qc_pulse']['freq_if']
    tiisq.setup()
    MC.settables([QCPulseLengthParameter(tiisq._qrm, tiisq._qcm), QCPulseGainParameter(tiisq._qcm)])
    setpoints_length = np.arange(1,400,2)
    setpoints_gain = np.arange(0,20,1)
    MC.setpoints_grid([setpoints_length,setpoints_gain])
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.on()
    dataset = MC.run('Rabi Pulse Length and Gain', soft_avg = tiisq._general_settings['software_averages'])
    # Analyse data to look for the smallest qc_pulse length that renders off-resonance amplitude, determine corresponding pi_pulse gain
    # platform.pi_pulse_length = 
    # platform.pi_pulse_gain = 

def run_Rabi_pulse_length_and_amplitude():
    tiisq.load_settings()
    tiisq._LO_QRM_settings['frequency'] = tiisq._general_settings['resonator_freq'] - tiisq._QRM_settings['pulses']['ro_pulse']['freq_if']
    tiisq._LO_QCM_settings['frequency'] = tiisq._general_settings['qubit_freq'] + tiisq._QCM_settings['pulses']['qc_pulse']['freq_if']
    tiisq.setup()
    MC.settables([QCPulseLengthParameter(tiisq._qrm, tiisq._qcm), QCPulseAmplitudeParameter(tiisq._qcm)])
    setpoints_length = np.arange(1,1000,2)
    setpoints_amplitude = np.arange(0,100,2)
    MC.setpoints_grid([setpoints_length,setpoints_amplitude])
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.on()
    dataset = MC.run('Rabi Pulse Length and Gain', soft_avg = tiisq._general_settings['software_averages'])
    # Analyse data to look for the smallest qc_pulse length that renders off-resonance amplitude, determine corresponding pi_pulse gain
    # platform.pi_pulse_length = 
    # platform.pi_pulse_gain = 

def run_shifted_resonator_spectroscopy():
    pass

# T1: RX(pi) - wait t(rotates z) - readout
def run_t1():
    tiisq.load_settings()
    tiisq._LO_QRM_settings['frequency'] = tiisq._general_settings['resonator_freq'] - tiisq._QRM_settings['pulses']['ro_pulse']['freq_if']
    tiisq._LO_QCM_settings['frequency'] = tiisq._general_settings['qubit_freq'] + tiisq._QCM_settings['pulses']['qc_pulse']['freq_if']
    tiisq._QCM_settings['pulses']['qc_pulse']['length'] = tiisq._general_settings['pi_pulse_length']
    tiisq._QRM_settings['pulses']['ro_pulse']['start']= tiisq._QCM_settings['pulses']['qc_pulse']['length'] + 4
    tiisq._QCM_settings['gain'] = tiisq._general_settings['pi_pulse_gain']
    tiisq.setup()
    MC.settables(T1WaitParameter(tiisq._qrm, tiisq._qcm))
    MC.setpoints(np.arange(0,4500,30))
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.on()
    dataset = MC.run('T1', soft_avg = tiisq._general_settings['software_averages'])
    tiisq.stop()
    # fit data and determine T1
    # platform.t1 = 

# Ramsey: RX(pi/2) - wait t(rotates z) - RX(pi/2) - readout
def run_ramsey():
    tiisq.load_settings()
    tiisq._LO_QRM_settings['frequency'] = tiisq._general_settings['resonator_freq'] - tiisq._QRM_settings['pulses']['ro_pulse']['freq_if']
    tiisq._LO_QCM_settings['frequency'] = tiisq._general_settings['qubit_freq'] + tiisq._QCM_settings['pulses']['qc_pulse']['freq_if']
    tiisq._QCM_settings['gain'] = tiisq._general_settings['pi_pulse_gain']
    tiisq._QCM_settings['pulses'] = {
            'qc_pulse':{	"freq_if": 200e6,
                        "amplitude": tiisq._general_settings['pi_pulse_amplitude'],
                        "start": 0,  
                        "length": tiisq._general_settings['pi_pulse_length']//2,    
                        "offset_i": 0,
                        "offset_q": 0,
                        "shape": "Gaussian",
                        },
            'qc2_pulse':{	"freq_if": 200e6,
                        "amplitude": tiisq._general_settings['pi_pulse_amplitude'],
                        "start": tiisq._general_settings['pi_pulse_length']//2 + 0,  
                        "length": tiisq._general_settings['pi_pulse_length']//2,
                        "offset_i": 0,
                        "offset_q": 0,
                        "shape": "Gaussian",
                        }
    }
    tiisq.setup()
    MC.settables(RamseyWaitParameter(tiisq._qrm, tiisq._qcm))
    MC.setpoints(np.arange(4,1000,4))
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    print(tiisq._general_settings['pi_pulse_length']//2)
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.on()
    dataset = MC.run('Ramsey', soft_avg = tiisq._general_settings['software_averages'])
    tiisq.stop()
    # fit data and determine Ramsey Time and dephasing
    # platform.ramsey = 
    # platform.qubit_freq += dephasing

# Spin Echo: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - readout
def run_spin_echo():
    tiisq.load_settings()
    tiisq._LO_QRM_settings['frequency'] = tiisq._general_settings['resonator_freq'] - tiisq._QRM_settings['pulses']['ro_pulse']['freq_if']
    tiisq._LO_QCM_settings['frequency'] = tiisq._general_settings['qubit_freq'] + tiisq._QCM_settings['pulses']['qc_pulse']['freq_if']
    tiisq._QCM_settings['gain'] = tiisq._general_settings['pi_pulse_gain'],
    tiisq._QCM_settings['pulses'] = {
            'qc_pulse':{	"freq_if": 200e6,
                        "amplitude": tiisq._general_settings['pi_pulse_amplitude'],
                        "start": 0,  
                        "length": tiisq._general_settings['pi_pulse_length']//2,    
                        "offset_i": 0,
                        "offset_q": 0,
                        "shape": "Gaussian",
                        },
            'qc2_pulse':{	"freq_if": 200e6,
                        "amplitude": tiisq._general_settings['pi_pulse_amplitude'],    
                        "start": tiisq._general_settings['pi_pulse_length']//2 + 0,  
                        "length": tiisq._general_settings['pi_pulse_length'],
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
    dataset = MC.run('Spin Echo', soft_avg = tiisq._general_settings['software_averages'])
    tiisq.stop()
    # ?

# Spin Echo 3 Pulses: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - RX(pi/2) - readout
def run_spin_echo_3pulses():
    tiisq.load_settings()
    tiisq._LO_QRM_settings['frequency'] = tiisq._general_settings['resonator_freq'] - tiisq._QRM_settings['pulses']['ro_pulse']['freq_if']
    tiisq._LO_QCM_settings['frequency'] = tiisq._general_settings['qubit_freq'] + tiisq._QCM_settings['pulses']['qc_pulse']['freq_if']
    tiisq._QCM_settings['gain'] = tiisq._general_settings['pi_pulse_gain'],
    tiisq._QCM_settings['pulses'] = {
            'qc_pulse':{	"freq_if": 200e6,
                        "amplitude": tiisq._general_settings['pi_pulse_amplitude'],
                        "start": 0,  
                        "length": tiisq._general_settings['pi_pulse_length']//2,    
                        "offset_i": 0,
                        "offset_q": 0,
                        "shape": "Gaussian",
                        },
            'qc2_pulse':{	"freq_if": 200e6,
                        "amplitude": tiisq._general_settings['pi_pulse_amplitude'],    
                        "start": tiisq._general_settings['pi_pulse_length']//2 + 0,  
                        "length": tiisq._general_settings['pi_pulse_length'],
                        "offset_i": 0,
                        "offset_q": 0,
                        "shape": "Gaussian",
                        },
            'qc3_pulse':{	"freq_if": 200e6,
                        "amplitude": tiisq._general_settings['pi_pulse_amplitude'],    
                        "start": (3 * tiisq._general_settings['pi_pulse_length'])//2 + 0,  
                        "length": tiisq._general_settings['pi_pulse_length']//2,
                        "offset_i": 0,
                        "offset_q": 0,
                        "shape": "Gaussian",
                        }
    }
    tiisq.setup()
    MC.settables(SpinEcho3PWaitParameter(tiisq._qrm, tiisq._qcm))
    MC.setpoints(np.arange(4,500,10))
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq._LO_qrm.on()
    tiisq._LO_qcm.on()
    dataset = MC.run('Spin Echo', soft_avg = tiisq._general_settings['software_averages'])
    tiisq.stop()
    # ?

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
    #, LO_qcm: SGS100A):
        self._qcm = qcm
        #self._LO_qcm = LO_qcm
        
    def set(self,value):
        gain = value / 100
        self._qcm._settings['gain'] = gain
        #self._LO_qcm._settings['power'] = -1

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
    
    def __init__(self, qrm: Pulsar_QRM, qcm: Pulsar_QCM):
        self._qrm = qrm
        self._qcm = qcm
    def set(self,value):
        #must be >= 4ns <= 65535
        self._qrm._settings['pulses']['ro_pulse']['start'] = self._qcm._settings['pulses']['qc_pulse']['length'] + 4 + value
    
class RamseyWaitParameter():
    label = 'Time'
    unit = 'ns'
    name = 'ramsey_wait'
    initial_value = 0
    
    def __init__(self, qrm: Pulsar_QRM, qcm: Pulsar_QCM):
        self._qrm = qrm
        self._qcm = qcm
        
    def set(self,value):
        self._qcm._settings['pulses']['qc2_pulse']['start'] = tiisq._general_settings['pi_pulse_length']//2 + value
        self._qrm._settings['pulses']['ro_pulse']['start']= tiisq._general_settings['pi_pulse_length'] + value + 4
        
class SpinEchoWaitParameter():
    label = 'Time'
    unit = 'ns'
    name = 'spin_echo_wait'
    initial_value = 0
    
    def __init__(self, qrm: Pulsar_QRM, qcm: Pulsar_QCM):
        self._qrm = qrm
        self._qcm = qcm
        
    def set(self,value):
        self._qcm._settings['pulses']['qc2_pulse']['start'] = tiisq._general_settings['pi_pulse_length']//2 + value
        self._qrm._settings['pulses']['ro_pulse']['start']= 3 * tiisq._general_settings['pi_pulse_length']//2 + 2 * value + 4

class SpinEcho3PWaitParameter():
    label = 'Time'
    unit = 'ns'
    name = 'spin_echo_wait'
    initial_value = 0
    
    def __init__(self, qrm: Pulsar_QRM, qcm: Pulsar_QCM):
        self._qrm = qrm
        self._qcm = qcm
        
    def set(self,value):
        self._qcm._settings['pulses']['qc2_pulse']['start'] = tiisq._general_settings['pi_pulse_length']//2 + value
        self._qcm._settings['pulses']['qc3_pulse']['start'] = (3 * tiisq._general_settings['pi_pulse_length'])//2 + 2 * value
        self._qrm._settings['pulses']['ro_pulse']['start']= 2 * tiisq._general_settings['pi_pulse_length'] + 2 * value + 4
