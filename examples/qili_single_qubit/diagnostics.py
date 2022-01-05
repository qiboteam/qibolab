import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from qibolab import pulses
from qibolab import platform
from qibolab.pulse_shapes import Rectangular, Gaussian

# TODO: Have a look in the documentation of ``MeasurementControl``
from quantify_core.measurement import MeasurementControl
from quantify_core.measurement.control import Gettable, Settable
from quantify_core.data.handling import set_datadir
from quantify_core.visualization.pyqt_plotmon import PlotMonitor_pyqt
from quantify_core.visualization.instrument_monitor import InstrumentMonitor
# TODO: Check why this set_datadir is needed
set_datadir(pathlib.Path(__file__).parent / "data")

#Fitting
import lmfit
from quantify_core.analysis.base_analysis import BaseAnalysis
import scipy as sc
from scipy.fft import fft, ifft
import glob
import os.path
import scipy.optimize as scopt
import math as mt

def create_measurement_control(name):
    import os
    #from quantify_core.measurement import MeasurementControl
    mc = MeasurementControl(f'MC {name}')
    #from quantify_core.visualization.pyqt_plotmon import PlotMonitor_pyqt
    plotmon = PlotMonitor_pyqt(f'Plot Monitor {name}')
    #from quantify_core.visualization.instrument_monitor import InstrumentMonitor
    insmon = InstrumentMonitor(f"Instruments Monitor {name}")        
    plotmon.tuids_max_num(3)
    mc.instr_plotmon(plotmon.name)
    mc.instrument_monitor(insmon.name)
    return mc, plotmon, insmon


class ROController():
    # Quantify Gettable Interface Implementation
    label = ['Amplitude', 'Phase','I','Q']
    unit = ['V', 'Radians','V','V']
    name = ['A', 'Phi','I','Q']

    def __init__(self, sequence):
        self.sequence = sequence

    def get(self):
        return platform(self.sequence)


def scanrange(width, step):
    scanrange = np.arange(-width,width,step)
    return scanrange

def asymetric_scanrange(width, step):
    scanrange = np.arange(-width,width/3,step)
    return scanrange

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


def get_pulse_sequence(duration=150, ro_amplitude=0.4, qc_amplitude=0.6):
    qc_pulse = pulses.Pulse(start=0,
                            frequency=100e6,
                            amplitude=qc_amplitude,
                            duration=duration,
                            phase=0,
                            shape=Gaussian(4000 / 5))
    ro_pulse = pulses.ReadoutPulse(start=duration + 4,
                                   frequency=20e6,
                                   amplitude=ro_amplitude,
                                   duration=2000,
                                   phase=0,
                                   shape=Rectangular())
    sequence = pulses.PulseSequence()
    sequence.add(qc_pulse)
    sequence.add(ro_pulse)
    return sequence, qc_pulse, ro_pulse


def run_resonator_spectroscopy(mc, width, step):
    sequence, qc_pulse, ro_pulse = get_pulse_sequence()
    
    # Fast Sweep
    platform.software_averages = 1
    scanrange = scanrange(width, step)
    #mc, pl, ins = create_measurement_control('resonator_spectroscopy')
    mc.settables(platform.LO_qrm.device.frequency)
    mc.setpoints(scanrange + platform.LO_qrm.get_frequency())
    mc.gettables(Gettable(ROController(sequence)))
    platform.LO_qrm.on()
    platform.LO_qcm.off()
    dataset = mc.run("Resonator Spectroscopy Fast", soft_avg=platform.software_averages)
    # http://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html
    platform.stop()
    platform.LO_qrm.set_frequency(dataset['x0'].values[dataset['y0'].argmax().values]) #here it changes the frequency

    
    resonator_freq = dataset['x0'].values[dataset['y0'].argmax()] + ro_pulse.frequency 
    print(f"\nResonator Frequency = {resonator_freq}")
    print(f"\nResonator LO Frequency  = {resonator_freq - ro_pulse.frequency}")
    fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
    ax.plot(dataset['x0'].values/1e9, dataset['y0'].values*1e3,'-',color='C0')
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')
    
    #small sweep
    platform.software_averages = 1
    scanrange = variable_resolution_scanrange(6e6, 1e6, 2e6, 0.1e6)#(25e6, 2e6, 5e6, 0.5e6)
    mc.settables(platform.LO_qrm.device.frequency) #(resonator_freq)
    mc.setpoints(scanrange + platform.LO_qrm.get_frequency())
    mc.gettables(Gettable(ROController(sequence)))
    platform.LO_qrm.on()
    platform.LO_qcm.off()
    dataset = mc.run("Resonator Spectroscopy precision", soft_avg=platform.software_averages)
    # http://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html
    platform.stop()
    
    resonator_freq = dataset['x0'].values[dataset['y0'].argmax()] + ro_pulse.frequency
    print(f"\nResonator Frequency = {resonator_freq}")
    print(f"\nResonator LO Frequency  = {resonator_freq - ro_pulse.frequency}")
    fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
    ax.plot(dataset['x0'].values/1e9, dataset['y0'].values*1e3,'-',color='C0')
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')


    return resonator_freq, dataset

def run_punchout(mc, resonator_freq, width, step):
    sequence, qc_pulse, ro_pulse = get_pulse_sequence(ro_amplitude=1)
    
    #-30dBm to 3.98dBm = 0.02V to 1V
    #Amplitude should be 1V then gain from 0.02 to 1 so values go from 0.02V to 1V so -30dBm to 3.98dBm
    #The range in the punchout is from 5 to -20dBm in VNA - 30dB from the attenuator. So, from -50 to -25dBm. So, from 0.002V to 0.035V.
    #5 to -20dBm = 1.125 to 0.063V.
    #get_pulse_sequence defines default amplitude to 0.4 and default pulse duration to 2000.
    #I see no color difference between gain 0 and gain 100.

    # Fast Sweep
    platform.software_averages = 1
    scanrange = scanrange(width, step)
    scanrange = (scanrange + (resonator_freq - ro_pulse.frequency))

    mc.settables([Settable(platform.LO_qrm.device.frequency),
                  Settable(QRPulseGainParameter(platform.qrm))])
    setpoints_gain = np.arange(0, 100, 10)
    mc.setpoints_grid([scanrange, setpoints_gain])
    mc.gettables(Gettable(ROController(sequence)))
    platform.LO_qrm.on()
    platform.LO_qcm.off()
    dataset = mc.run("Punchout", soft_avg=platform.software_averages)
    # http://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html
    platform.stop()
    platform.LO_qrm.set_frequency(dataset['x0'].values[dataset['y0'].argmax().values]) #here it changes the frequency
    resonator_freq = dataset['x0'].values[dataset['y0'].argmax()] + ro_pulse.frequency 
    print(f"\nResonator Frequency = {resonator_freq}")
    print(f"\nResonator LO Frequency  = {resonator_freq - ro_pulse.frequency}")
    fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
    ax.plot(dataset['x0'].values/1e9, dataset['y0'].values*1e3,'-',color='C0')
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')

    return resonator_freq, dataset


def run_qubit_spectroscopy(mc, resonator_freq, width, step):
    sequence, qc_pulse, ro_pulse = get_pulse_sequence()

    #platform.LO_qcm.set_frequency(fq)

    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    # Fast Sweep
    platform.software_averages = 1
    scanrange = scanrange(width, step)
    mc.settables(platform.LO_qcm.device.frequency)
    mc.setpoints(scanrange + platform.LO_qcm.get_frequency())
    mc.gettables(Gettable(ROController(sequence)))
    platform.LO_qrm.on()
    platform.LO_qcm.on()
    dataset = mc.run("Qubit Spectroscopy Fast", soft_avg=platform.software_averages)
    platform.stop()
    # http://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html
    platform.LO_qcm.set_frequency(dataset['x0'].values[dataset['y0'].argmin().values])


    qubit_freq = dataset['x0'].values[dataset['y0'].argmin()] - qc_pulse.frequency 
    print(f"\nQubit Frequency = {qubit_freq}")
    print(f"\nQubit LO Frequency  = {qubit_freq + qc_pulse.frequency}")
    fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
    ax.plot(dataset['x0'].values/1e9, dataset['y0'].values*1e3,'-',color='C0')
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')
    
    #small sweep
    platform.software_averages = 4 
    scanrange = variable_resolution_scanrange(20e6, 1e6, 3e6, 0.2e6)#(25e6, 2e6, 5e6, 0.5e6)
    mc.settables(platform.LO_qcm.device.frequency) #(resonator_freq)
    mc.setpoints(scanrange + platform.LO_qcm.get_frequency())
    mc.gettables(Gettable(ROController(sequence)))
    platform.LO_qrm.on()
    platform.LO_qcm.on()
    dataset = mc.run("Qubit Spectroscopy precision", soft_avg=platform.software_averages)
    # http://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html
    platform.stop()
    
    qubit_freq = dataset['x0'].values[dataset['y0'].argmin()] - qc_pulse.frequency
    print(f"\nQubit Frequency = {qubit_freq}")
    print(f"\nQubit LO Frequency  = {qubit_freq + qc_pulse.frequency}")
    fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
    ax.plot(dataset['x0'].values/1e9, dataset['y0'].values*1e3,'-',color='C0')
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')

    return qubit_freq, dataset

def run_qubit_spectroscopy_gain(mc, resonator_freq, qubit_freq):
    sequence, qc_pulse, ro_pulse = get_pulse_sequence(qc_amplitude=1)


    platform.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)

    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    # Fast Sweep
    platform.software_averages = 4
    scanrange = asymetric_scanrange(20e6,1e6) #variable scanrange doesn't work with grid.
    scanrange = (scanrange + platform.LO_qcm.get_frequency())

    mc.settables([Settable(platform.LO_qcm.device.frequency),
                  Settable(QCPulseGainParameter(platform.qcm))])
    setpoints_gain = np.arange(10, 50, 20)
    mc.setpoints_grid([scanrange, setpoints_gain])
    mc.gettables(Gettable(ROController(sequence)))
    platform.LO_qrm.on()
    platform.LO_qcm.on()
    dataset = mc.run("Qubit Spectroscopy Gain", soft_avg=platform.software_averages)
    platform.stop()
    # http://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html
    platform.LO_qcm.set_frequency(dataset['x0'].values[dataset['y0'].argmin().values])


    qubit_freq = dataset['x0'].values[dataset['y0'].argmin()] - qc_pulse.frequency 
    print(f"\nQubit Frequency = {qubit_freq}")
    print(f"\nQubit LO Frequency  = {qubit_freq + qc_pulse.frequency}")
    fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
    ax.plot(dataset['x0'].values/1e9, dataset['y0'].values*1e3,'-',color='C0')
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')


    return qubit_freq, dataset

def run_rabi_pulse_length(resonator_freq, qubit_freq):
    sequence, qc_pulse, ro_pulse = get_pulse_sequence()

    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    platform.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)
    mc, pl, ins = create_measurement_control('Rabi_pulse_length')
    platform.software_averages = 3
    mc.settables(Settable(QCPulseLengthParameter(ro_pulse, qc_pulse)))
    mc.setpoints(np.arange(1, 1200, 10))
    mc.gettables(Gettable(ROController(sequence)))
    platform.LO_qrm.on()
    platform.LO_qcm.on()
    dataset = mc.run('Rabi Pulse Length', soft_avg = platform.software_averages)
    platform.stop()


def run_rabi_pulse_gain(resonator_freq, qubit_freq):
    sequence, qc_pulse, ro_pulse = get_pulse_sequence(duration=200)

    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    platform.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)
    mc, pl, ins = create_measurement_control('Rabi_pulse_gain')
    platform.software_averages = 1
    mc.settables(Settable(QCPulseGainParameter(platform.qcm)))
    mc.setpoints(np.arange(0, 100))
    mc.gettables(Gettable(ROController(sequence)))
    platform.LO_qrm.on()
    platform.LO_qcm.on()
    dataset = mc.run('Rabi Pulse Gain', soft_avg = platform.software_averages)
    platform.stop()


def run_rabi_pulse_length_and_gain(resonator_freq, qubit_freq):
    sequence, qc_pulse, ro_pulse = get_pulse_sequence()

    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    platform.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)
    mc, pl, ins = create_measurement_control('Rabi_pulse_length_and_gain')
    platform.software_averages = 1
    mc.settables([Settable(QCPulseLengthParameter(ro_pulse, qc_pulse)),
                  Settable(QCPulseGainParameter(platform.qcm))])
    setpoints_length = np.arange(1, 400, 10)
    setpoints_gain = np.arange(0, 20, 1)
    mc.setpoints_grid([setpoints_length, setpoints_gain])
    mc.gettables(Gettable(ROController(sequence)))
    platform.LO_qrm.on()
    platform.LO_qcm.on()
    dataset = mc.run('Rabi Pulse Length and Gain', soft_avg = platform.software_averages)
    # Analyse data to look for the smallest qc_pulse length that renders off-resonance amplitude, determine corresponding pi_pulse gain
    # platform.pi_pulse_length =
    # platform.pi_pulse_gain =
    platform.stop()


def run_rabi_pulse_length_and_amplitude(resonator_freq, qubit_freq):
    sequence, qc_pulse, ro_pulse = get_pulse_sequence()

    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    platform.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)
    mc, pl, ins = create_measurement_control('Rabi_pulse_length_and_amplitude')
    platform.software_averages = 1
    mc.settables([Settable(QCPulseLengthParameter(ro_pulse, qc_pulse)),
                  Settable(QCPulseAmplitudeParameter(qc_pulse))])
    setpoints_length = np.arange(1, 1000, 2)
    setpoints_amplitude = np.arange(0, 100, 2)
    mc.setpoints_grid([setpoints_length, setpoints_amplitude])
    mc.gettables(Gettable(ROController(sequence)))
    platform.LO_qrm.on()
    platform.LO_qcm.on()
    dataset = mc.run('Rabi Pulse Length and Gain', soft_avg = platform.software_averages)
    # Analyse data to look for the smallest qc_pulse length that renders off-resonance amplitude, determine corresponding pi_pulse gain
    # platform.pi_pulse_length =
    # platform.pi_pulse_gain =
    platform.stop()


def run_t1(resonator_freq, qubit_freq, pi_pulse_gain, pi_pulse_length,
            delay_before_readout_start, delay_before_readout_end,
            delay_before_readout_step):
    sequence, qc_pulse, ro_pulse = get_pulse_sequence(duration=pi_pulse_length)

    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    platform.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)
    platform.qcm.gain = pi_pulse_gain

    mc, pl, ins = create_measurement_control('t1')
    mc.settables(Settable(T1WaitParameter(ro_pulse, qc_pulse)))
    mc.setpoints(np.arange(delay_before_readout_start,
                           delay_before_readout_end,
                           delay_before_readout_step))
    mc.gettables(Gettable(ROController(sequence)))
    platform.LO_qrm.on()
    platform.LO_qcm.on()
    dataset = mc.run('T1', soft_avg = platform.software_averages)
    platform.stop()
    # fit data and determine T1
    # platform.t1 =

    return dataset


def run_ramsey(resonator_freq, qubit_freq, pi_pulse_gain, pi_pulse_length, pi_pulse_amplitude,
               start_start, start_end, start_step):
    qc_pulse = pulses.Pulse(start=0,
                            frequency=200000000.0,
                            amplitude=pi_pulse_amplitude,
                            duration=pi_pulse_length // 2,
                            phase=0,
                            shape=Gaussian(pi_pulse_length // 10))
    qc2_pulse = pulses.Pulse(start=pi_pulse_length // 2 + 0,
                               frequency=200000000.0,
                               amplitude=pi_pulse_amplitude,
                               duration=pi_pulse_length // 2,
                               phase=0,
                               shape=Gaussian(pi_pulse_length // 10))
    start = qc_pulse.duration + qc2_pulse.duration + 4
    ro_pulse = pulses.ReadoutPulse(start=start,
                                   frequency=20000000.0,
                                   amplitude=0.9,
                                   duration=2000,
                                   phase=0,
                                   shape=Rectangular())
    sequence = pulses.PulseSequence()
    sequence.add(qc_pulse)
    sequence.add(qc2_pulse)
    sequence.add(ro_pulse)

    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    platform.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)
    platform.qcm.gain = pi_pulse_gain

    mc, pl, ins = create_measurement_control('ramsey')
    mc.settables(Settable(RamseyWaitParameter(ro_pulse, qc2_pulse, pi_pulse_length)))
    mc.setpoints(np.arange(start_start, start_end, start_step))
    mc.gettables(Gettable(ROController(sequence)))
    platform.LO_qrm.on()
    platform.LO_qcm.on()
    dataset = mc.run('Ramsey', soft_avg = platform.software_averages)
    platform.stop()
    # fit data and determine Ramsey Time and dephasing
    # platform.ramsey =
    # platform.qubit_freq += dephasing
    return dataset


def run_spin_echo(resonator_freq, qubit_freq, pi_pulse_gain, pi_pulse_length, pi_pulse_amplitude,
                  start_start, start_end, start_step):
    qc_pulse = pulses.Pulse(start=0,
                            frequency=200000000.0,
                            amplitude=pi_pulse_amplitude,
                            duration=pi_pulse_length // 2,
                            phase=0,
                            shape=Gaussian(pi_pulse_length // 10))
    qc2_pulse = pulses.Pulse(start=pi_pulse_length // 2 + 0,
                             frequency=200000000.0,
                             amplitude=pi_pulse_amplitude,
                             duration=pi_pulse_length // 2,
                             phase=0,
                             shape=Gaussian(pi_pulse_length // 10))
    start = qc_pulse.duration + qc2_pulse.duration + 4
    ro_pulse = pulses.ReadoutPulse(start=start,
                                   frequency=20000000.0,
                                   amplitude=0.9,
                                   duration=2000,
                                   phase=0,
                                   shape=Rectangular())
    sequence = pulses.PulseSequence()
    sequence.add(qc_pulse)
    sequence.add(qc2_pulse)
    sequence.add(ro_pulse)

    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    platform.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)
    platform.qcm.gain = pi_pulse_gain

    mc, pl, ins = create_measurement_control('spin_echo')
    mc.settables(Settable(SpinEchoWaitParameter(ro_pulse, qc2_pulse, pi_pulse_length)))
    mc.setpoints(np.arange(start_start, start_end, start_step))
    mc.gettables(Gettable(ROController(sequence)))
    platform.LO_qrm.on()
    platform.LO_qcm.on()
    dataset = mc.run('Spin Echo', soft_avg = platform.software_averages)
    platform.stop()

    return dataset


# help classes

class QCPulseLengthParameter():

    label = 'Qubit Control Pulse Length'
    unit = 'ns'
    name = 'qc_pulse_length'

    def __init__(self, ro_pulse, qc_pulse):
        self.ro_pulse = ro_pulse
        self.qc_pulse = qc_pulse

    def set(self, value):
        self.qc_pulse.duration = value
        self.ro_pulse.start = value + 4


class QCPulseGainParameter():

    label = 'Qubit Control Gain'
    unit = '%'
    name = 'qc_pulse_gain'

    def __init__(self, qcm):
        self.qcm = qcm

    def set(self,value):
        self.qcm.gain = value / 100

class QCPulseAmplitudeParameter():

    label = 'Qubit Control Pulse Amplitude'
    unit = '%'
    name = 'qc_pulse_amplitude'

    def __init__(self, qc_pulse):
        self.qc_pulse = qc_pulse

    def set(self, value):
        self.qc_pulse.amplitude = value / 100


class T1WaitParameter():
    label = 'Time'
    unit = 'ns'
    name = 't1_wait'
    initial_value = 0

    def __init__(self, ro_pulse, qc_pulse):
        self.ro_pulse = ro_pulse
        self.base_duration = qc_pulse.duration

    def set(self, value):
        # TODO: implement following condition
        #must be >= 4ns <= 65535
        #platform.delay_before_readout = value
        self.ro_pulse.start = self.base_duration + 4 + value


class RamseyWaitParameter():
    label = 'Time'
    unit = 'ns'
    name = 'ramsey_wait'
    initial_value = 0

    def __init__(self, ro_pulse, qc2_pulse, pi_pulse_length):
        self.ro_pulse = ro_pulse
        self.qc2_pulse = qc2_pulse
        self.pi_pulse_length = pi_pulse_length

    def set(self, value):
        self.qc2_pulse.start = self.pi_pulse_length // 2 + value
        self.ro_pulse.start = self.pi_pulse_length + value + 4


class SpinEchoWaitParameter():
    label = 'Time'
    unit = 'ns'
    name = 'spin_echo_wait'
    initial_value = 0

    def __init__(self, ro_pulse, qc2_pulse, pi_pulse_length):
        self.ro_pulse = ro_pulse
        self.qc2_pulse = qc2_pulse
        self.pi_pulse_length = pi_pulse_length

    def set(self, value):
        self.qc2_pulse.start = self.pi_pulse_length//2 + value
        self.ro_pulse.start = 3 * self.pi_pulse_length//2 + 2 * value + 4


#added DavidE 30/12/2021
class QRPulseGainParameter():

    label = 'Qubit Readout Gain'
    unit = '%'
    name = 'ro_pulse_gain'

    def __init__(self, qrm):
        self.qrm = qrm

    def set(self,value):
        self.qrm.gain = value / 100

# Fitting
"""
Author: David Eslava Sabaté (david.eslava@qilimanjaro.tech)
Last update: 29/12/2021
Qilimanjaro Quantum Tech
"""

def data_post():
    #get last measured file
    directory = '.\data'
    directory = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime)
    directory = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime)
    label = os.path.basename(os.path.normpath(directory))
    set_datadir('data')
    d = BaseAnalysis(tuid=label)
    d.run()
    data = d.dataset
    #clean the array 
    plt.plot(data.x0,data.y0)  
    arr1 = data.y0;      
    voltage = [None] * len(arr1);     
    for i in range(0, len(arr1)):    
        voltage[i] = float(arr1[i]);         
    arr1 = data.x0;        
    frequency = [None] * len(arr1);       
    for i in range(0, len(arr1)):    
        frequency[i] = float(arr1[i]/1e9); 
    return voltage, frequency, data, d


def Lorentzian_fit(voltage, frequency,peak, data):
    def resonator_peak(frequency,amplitude,center,sigma,offset):
        return (amplitude/np.pi) * (sigma/((frequency-center)**2 + sigma**2) + offset)
    model_Q = lmfit.Model(resonator_peak)

    #to guess center
    if peak == max:
        guess_center = frequency[np.argmax(voltage)] #Argmax = Returns the indices of the maximum values along an axis.
        model_Q.set_param_hint('center',value=guess_center,vary=True)
    else:
        guess_center = frequency[np.argmin(voltage)] #Argmin = Returns the indices of the minimum values along an axis.
        model_Q.set_param_hint('center',value=guess_center,vary=False)

    #to guess the sigma
    if peak == max:
        voltage_min_i = np.argmin(voltage)
        frequency_voltage_min = frequency[voltage_min_i]
        guess_sigma = abs(frequency_voltage_min - guess_center)#5e-04 #500KHz*1e-9
        model_Q.set_param_hint('sigma',value=guess_sigma,
                                vary=True)
    else: 
        guess_sigma = 5e-03 #500KHz*1e-9
        model_Q.set_param_hint('sigma',value=guess_sigma,
                                vary=True)
    #to guess the amplitude 
    #http://openafox.com/science/peak-function-derivations.html

    if peak == max:
        voltage_max = np.max(voltage)
        guess_amp = voltage_max*guess_sigma*np.pi
        model_Q.set_param_hint('amplitude',value=guess_amp,
                                vary=True)
    else:
        voltage_min = np.min(voltage)
        guess_amp = -voltage_min*guess_sigma*np.pi
        model_Q.set_param_hint('amplitude',value=guess_amp,
                                vary=True)
    #to guess the offset
    guess_offset = voltage[0]*-2.5*1e5
    model_Q.set_param_hint('offset',value=guess_offset,
                            vary=True)
    #guessed parameters
    guess_parameters = model_Q.make_params()
    guess_parameters

    #fit the model with the data and guessed parameters
    fit_res = model_Q.fit(data=voltage,frequency=frequency,params=guess_parameters)
    #print(fit_res.fit_report())
    #fit_res.best_values
    #get the values for postprocessing and for legend.
    f0 = fit_res.best_values['center']
    BW = fit_res.best_values['sigma']*2
    Q = abs(f0/BW)
    #plot the fitted curve
    dummy_frequencies = np.linspace(np.amin(frequency),np.amax(frequency),101)
    fit_fine = resonator_peak(dummy_frequencies,**fit_res.best_values)
    fig,ax = plt.subplots(1,1,figsize=(8,3))
    ax.plot(data.x0/1e9,data.y0*1e3,'o',label='Data')
    ax.plot(dummy_frequencies,fit_fine*1e3,'r-',
            label=r"Fit $f_0$ ={:.4f} GHz"
            "\n" "     $Q$ ={:.0f}".format(f0,Q))
    ax.set_ylabel('Integrated Voltage (mV)')
    ax.set_xlabel('Frequency (GHz)')
    ax.legend()
    fig.savefig('Resonator_spec.pdf',format='pdf')
    #fit_res.plot_fit(show_init=True)
    return f0, BW, Q

def qubit_frequency_guess (resonator_freq, resonator_freq_highP, g, alpha):
    resonator_freq = resonator_freq/1e9
    resonator_freq_highP = resonator_freq_highP/1e9
    g = (g/1e9)/ 2*np.pi
    alpha = alpha/1e9
    measured_chi = -(resonator_freq_highP-resonator_freq)
    def function(x):
        delta = (x[0]-resonator_freq)
        return (g**2*alpha/(delta*(delta-alpha)))-measured_chi
    fq = scopt.fsolve(function,[4.])
    print(f"Approximated qubit frequency is around {fq} GHz")
    return fq

def plot_punchout ():
    voltage, frequency, data, d = data_post()
    x_vec = d.dataset.x0[:d.dataset.xlen]
    y_vec = d.dataset.x1[::d.dataset.xlen]
    z_mat = np.array(d.dataset.y0).reshape((d.dataset.ylen,d.dataset.xlen))
    def plot_2d_uniform(x,y,z,ax,**kw):
        x_np = np.array(x)
        y_np = np.array(y)
        x_delta = x_np[1]-x_np[0]
        y_delta = y_np[1]-y_np[0]
        x_edges = np.concatenate((x_np-x_delta*0.5,[x_np[-1]+x_delta*0.5]))
        y_edges = np.concatenate((y_np-y_delta*0.5,[y_np[-1]+y_delta*0.5]))
        return ax.pcolormesh(x_edges,y_edges,z, **kw)
    ax = plt.gca()
    c = plot_2d_uniform(x_vec,y_vec,1e3*z_mat, ax=ax, cmap='viridis')
    fig = plt.gcf()
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Integrated Voltage (mV)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (a.u.)')
    ax.set_title('Punchout')
    fig.savefig('Punchout.pdf',format='pdf')