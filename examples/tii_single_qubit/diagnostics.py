import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from qibolab import pulses
from qibolab.platforms import TIIq

# TODO: Have a look in the documentation of ``MeasurementControl``
from quantify_core.measurement import MeasurementControl
from quantify_core.measurement.control import Gettable
from quantify_core.data.handling import set_datadir
# TODO: Check why this set_datadir is needed
set_datadir(pathlib.Path(__file__).parent / "data")


class ROController():
    # Quantify Gettable Interface Implementation
    label = ['Amplitude', 'Phase','I','Q']
    unit = ['V', 'Radians','V','V']
    name = ['A', 'Phi','I','Q']

    def __init__(self, qrm, qcm, qrm_sequence, qcm_sequence):
        self.qrm = qrm
        self.qcm = qcm
        self.qrm_sequence = qrm_sequence
        self.qcm_sequence = qcm_sequence

    def get(self):
        #self.qrm.setup(qrm._settings) # this has already been done earlier?
        waveforms, program = self.qrm.translate(self.qrm_sequence)
        self.qrm.upload(waveforms, program, "./data")

        #self.qcm.setup(qcm._settings)
        waveforms, program = self.qcm.translate(self.qcm_sequence)
        self.qcm.upload(waveforms, program, "./data")

        self.qcm.play_sequence()
        acquisition_results = self.qrm.play_sequence_and_acquire(self.qrm_sequence.readout_pulse)
        return acquisition_results


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


def run_resonator_spectroscopy(lowres_width, lowres_step,
                               highres_width, highres_step,
                               precision_width, precision_step):
    with open("tii_single_qubit_settings.json", "r") as file:
        settings = json.load(file)

    tiiq = TIIq()
    tiiq.setup(settings)

    ro_pulse = pulses.TIIReadoutPulse(name="ro_pulse",
                                      start=70,
                                      frequency=20000000.0,
                                      amplitude=0.5,
                                      length=3000,
                                      shape="Block",
                                      delay_before_readout=4)
    qc_pulse = pulses.TIIPulse(name="qc_pulse",
                               start=0,
                               frequency=200000000.0,
                               amplitude=0.3,
                               length=60,
                               shape="Gaussian")
    qrm_sequence = pulses.PulseSequence()
    qrm_sequence.add(ro_pulse)
    qcm_sequence = pulses.PulseSequence()
    qcm_sequence.add(qc_pulse)

    mc = MeasurementControl('MC_resonator_spectroscopy')
    # Fast Sweep
    tiiq.software_averages = 1
    scanrange = variable_resolution_scanrange(lowres_width, lowres_step, highres_width, highres_step)
    mc.settables(tiiq.LO_qrm.device.frequency)
    mc.setpoints(scanrange + tiiq.LO_qrm.get_frequency())
    mc.gettables(Gettable(ROController(tiiq.qrm, tiiq.qcm, qrm_sequence, qcm_sequence)))
    tiiq.LO_qrm.on()
    tiiq.LO_qcm.off()
    dataset = mc.run("Resonator Spectroscopy Fast", soft_avg=tiiq.software_averages)
    # http://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html
    tiiq.stop()
    tiiq.LO_qrm.set_frequency(dataset['x0'].values[dataset['y0'].argmax().values])

    # Precision Sweep
    tiiq.software_averages = 1 # 3
    scanrange = np.arange(-precision_width, precision_width, precision_step)
    mc.settables(tiiq.LO_qrm.device.frequency)
    mc.setpoints(scanrange + tiiq.LO_qrm.get_frequency())
    mc.gettables(Gettable(ROController(tiiq.qrm, tiiq.qcm, qrm_sequence, qcm_sequence)))
    tiiq.LO_qrm.on()
    tiiq.LO_qcm.off()
    dataset = mc.run("Resonator Spectroscopy Precision", soft_avg=tiiq.software_averages)
    tiiq.stop()

    from scipy.signal import savgol_filter
    smooth_dataset = savgol_filter(dataset['y0'].values, 25, 2)
    resonator_freq = dataset['x0'].values[smooth_dataset.argmax()] + ro_pulse.frequency
    print(f"\nResonator Frequency = {resonator_freq}")
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
    plt.savefig("run_resonator_spectroscopy.pdf")

    return resonator_freq, dataset


def run_qubit_spectroscopy(fast_start, fast_end, fast_step,
                           precision_start, precision_end, precision_step):
    with open("tii_single_qubit_settings.json", "r") as file:
        settings = json.load(file)

    tiiq = TIIq()
    tiiq.setup(settings)

    ro_pulse = pulses.TIIReadoutPulse(name="ro_pulse",
                                      start=70,
                                      frequency=20000000.0,
                                      amplitude=0.5,
                                      length=4040,
                                      shape="Block",
                                      delay_before_readout=4)
    qc_pulse = pulses.TIIPulse(name="qc_pulse",
                               start=0,
                               frequency=200000000.0,
                               amplitude=0.3,
                               length=4000,
                               shape="Gaussian")
    qrm_sequence = pulses.PulseSequence()
    qrm_sequence.add(ro_pulse)
    qcm_sequence = pulses.PulseSequence()
    qcm_sequence.add(qc_pulse)

    mc = MeasurementControl('MC_qubit_spectroscopy')
    # Fast Sweep
    tiiq.software_averages = 1
    scanrange = np.arange(fast_start, fast_end, fast_step)
    mc.settables(tiiq.LO_qcm.device.frequency)
    mc.setpoints(scanrange + tiiq.LO_qcm.get_frequency())
    mc.gettables(Gettable(ROController(tiiq.qrm, tiiq.qcm, qrm_sequence, qcm_sequence)))
    tiiq.LO_qrm.on()
    tiiq.LO_qcm.off()
    dataset = mc.run("Qubit Spectroscopy Fast", soft_avg=tiiq.software_averages)
    # http://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html
    tiiq.stop()
    tiiq.LO_qcm.set_frequency(dataset['x0'].values[dataset['y0'].argmin().values])

    # Precision Sweep
    tiiq.software_averages = 3
    scanrange = np.arange(precision_start, precision_end, precision_step)
    mc.settables(tiiq.LO_qcm.device.frequency)
    mc.setpoints(scanrange + tiiq.LO_qcm.get_frequency())
    mc.gettables(Gettable(ROController(tiiq.qrm, tiiq.qcm, qrm_sequence, qcm_sequence)))
    tiiq.LO_qrm.on()
    tiiq.LO_qcm.off()
    dataset = mc.run("Qubit Spectroscopy Precision", soft_avg=tiiq.software_averages)
    tiiq.stop()

    from scipy.signal import savgol_filter
    smooth_dataset = savgol_filter(dataset['y0'].values, 11, 2)
    qubit_freq = dataset['x0'].values[smooth_dataset.argmin()] - qc_pulse.frequency
    print(dataset['x0'].values[smooth_dataset.argmin()])
    print(f"Qubit Frequency = {qubit_freq}")
    print(len(dataset['y0'].values))
    print(len(smooth_dataset))

    fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
    ax.plot(dataset['x0'].values, dataset['y0'].values,'-',color='C0')
    ax.plot(dataset['x0'].values, smooth_dataset,'-',color='C1')
    ax.title.set_text('Original')
    #ax.xlabel("Frequency")
    #ax.ylabel("Amplitude")
    ax.plot(dataset['x0'].values[smooth_dataset.argmin()], smooth_dataset[smooth_dataset.argmin()], 'o', color='C2')
    plt.savefig("run_qubit_spectroscopy.pdf")

    return qubit_freq, dataset

def run_rabi_pulse_length(resonator_freq, qubit_freq):
    with open("tii_single_qubit_settings.json", "r") as file:
        settings = json.load(file)
    tiiq = TIIq()
    tiiq.setup(settings)
    ro_pulse = pulses.TIIReadoutPulse(name="ro_pulse",
                                      start=70,
                                      frequency=20000000.0,
                                      amplitude=0.5,
                                      length=3000,
                                      shape="Block",
                                      delay_before_readout=4)
    qc_pulse = pulses.TIIPulse(name="qc_pulse",
                               start=0,
                               frequency=200000000.0,
                               amplitude=0.3,
                               length=60,
                               shape="Gaussian")
    qrm_sequence = pulses.PulseSequence()
    qrm_sequence.add(ro_pulse)
    qcm_sequence = pulses.PulseSequence()
    qcm_sequence.add(qc_pulse)
    tiiq.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    tiiq.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)
    mc = MeasurementControl('MC_Rabi_pulse_length')
    tiiq.software_averages = 1
    mc.settables(QCPulseLengthParameter(ro_pulse, qc_pulse))
    mc.setpoints(np.arange(1, 2000, 5))
    mc.gettables(Gettable(ROController(tiiq.qrm, tiiq.qcm, qrm_sequence, qcm_sequence)))
    tiiq.LO_qrm.on()
    tiiq.LO_qcm.on()
    dataset = mc.run('Rabi Pulse Length', soft_avg = tiiq.software_averages)
    tiiq.stop()


def run_rabi_pulse_gain(resonator_freq, qubit_freq):
    with open("tii_single_qubit_settings.json", "r") as file:
        settings = json.load(file)
    tiiq = TIIq()
    tiiq.setup(settings)
    ro_pulse = pulses.TIIReadoutPulse(name="ro_pulse",
                                      start=70,
                                      frequency=20000000.0,
                                      amplitude=0.5,
                                      length=3000,
                                      shape="Block",
                                      delay_before_readout=4)
    qc_pulse = pulses.TIIPulse(name="qc_pulse",
                               start=0,
                               frequency=200000000.0,
                               amplitude=0.3,
                               length=60,
                               shape="Gaussian")
    qrm_sequence = pulses.PulseSequence()
    qrm_sequence.add(ro_pulse)
    qcm_sequence = pulses.PulseSequence()
    qcm_sequence.add(qc_pulse)
    tiiq.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    tiiq.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)
    mc = MeasurementControl('MC_Rabi_pulse_gain')
    tiiq.software_averages = 1
    mc.settables(QCPulseGainParameter(tiiq.qcm))
    mc.setpoints(np.arange(0, 1, 0.02))
    mc.gettables(Gettable(ROController(tiiq.qrm, tiiq.qcm, qrm_sequence, qcm_sequence)))
    tiiq.LO_qrm.on()
    tiiq.LO_qcm.on()
    dataset = mc.run('Rabi Pulse Gain', soft_avg = tiiq.software_averages)
    tiiq.stop()


def run_rabi_pulse_length_and_gain(resonator_freq, qubit_freq):
    with open("tii_single_qubit_settings.json", "r") as file:
        settings = json.load(file)
    tiiq = TIIq()
    tiiq.setup(settings)
    ro_pulse = pulses.TIIReadoutPulse(name="ro_pulse",
                                      start=70,
                                      frequency=20000000.0,
                                      amplitude=0.5,
                                      length=3000,
                                      shape="Block",
                                      delay_before_readout=4)
    qc_pulse = pulses.TIIPulse(name="qc_pulse",
                               start=0,
                               frequency=200000000.0,
                               amplitude=0.3,
                               length=60,
                               shape="Gaussian")
    qrm_sequence = pulses.PulseSequence()
    qrm_sequence.add(ro_pulse)
    qcm_sequence = pulses.PulseSequence()
    qcm_sequence.add(qc_pulse)
    tiiq.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    tiiq.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)
    mc = MeasurementControl('MC_Rabi_pulse_length_and_gain')
    tiiq.software_averages = 1
    mc.settables([QCPulseLengthParameter(ro_pulse, qc_pulse), QCPulseGainParameter(tiiq.qcm)])
    setpoints_length = np.arange(1, 200, 10)
    setpoints_gain = np.arange(0, 100, 5)
    mc.setpoints_grid([setpoints_length, setpoints_gain])
    mc.gettables(Gettable(ROController(tiiq.qrm, tiiq.qcm, qrm_sequence, qcm_sequence)))
    tiiq.LO_qrm.on()
    tiiq.LO_qcm.on()
    dataset = mc.run('Rabi Pulse Length and Gain', soft_avg = tiiq.software_averages)
    # Analyse data to look for the smallest qc_pulse length that renders off-resonance amplitude, determine corresponding pi_pulse gain
    # platform.pi_pulse_length =
    # platform.pi_pulse_gain =
    tiiq.stop()


def run_rabi_pulse_length_and_amplitude(resonator_freq, qubit_freq):
    with open("tii_single_qubit_settings.json", "r") as file:
        settings = json.load(file)
    tiiq = TIIq()
    tiiq.setup(settings)
    ro_pulse = pulses.TIIReadoutPulse(name="ro_pulse",
                                      start=70,
                                      frequency=20000000.0,
                                      amplitude=0.5,
                                      length=3000,
                                      shape="Block",
                                      delay_before_readout=4)
    qc_pulse = pulses.TIIPulse(name="qc_pulse",
                               start=0,
                               frequency=200000000.0,
                               amplitude=0.3,
                               length=60,
                               shape="Gaussian")
    qrm_sequence = pulses.PulseSequence()
    qrm_sequence.add(ro_pulse)
    qcm_sequence = pulses.PulseSequence()
    qcm_sequence.add(qc_pulse)
    tiiq.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    tiiq.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)
    mc = MeasurementControl('MC_Rabi_pulse_length_and_amplitude')
    tiiq.software_averages = 1
    mc.settables([QCPulseLengthParameter(ro_pulse, qc_pulse), QCPulseAmplitudeParameter(qc_pulse)])
    setpoints_length = np.arange(1, 200, 10)
    setpoints_amplitude = np.arange(0, 100, 5)
    mc.setpoints_grid([setpoints_length, setpoints_amplitude])
    mc.gettables(Gettable(ROController(tiiq.qrm, tiiq.qcm, qrm_sequence, qcm_sequence)))
    tiiq.LO_qrm.on()
    tiiq.LO_qcm.on()
    dataset = mc.run('Rabi Pulse Length and Gain', soft_avg = tiiq.software_averages)
    # Analyse data to look for the smallest qc_pulse length that renders off-resonance amplitude, determine corresponding pi_pulse gain
    # platform.pi_pulse_length =
    # platform.pi_pulse_gain =
    tiiq.stop()


def run_t1(resonator_freq, qubit_freq, pi_pulse_gain, pi_pulse_length,
            delay_before_readout_start, delay_before_readout_end,
            delay_before_readout_step):
    with open("tii_single_qubit_settings.json", "r") as file:
        settings = json.load(file)

    tiiq = TIIq()
    settings['_QCM_settings']['gain'] = pi_pulse_gain
    tiiq.setup(settings)

    ro_pulse = pulses.TIIReadoutPulse(name="ro_pulse",
                                      start=70,
                                      frequency=20000000.0,
                                      amplitude=0.5,
                                      length=3000,
                                      shape="Block",
                                      delay_before_readout=4)
    qc_pulse = pulses.TIIPulse(name="qc_pulse",
                               start=0,
                               frequency=200000000.0,
                               amplitude=0.3,
                               length=pi_pulse_length,
                               shape="Gaussian")
    qrm_sequence = pulses.PulseSequence()
    qrm_sequence.add(ro_pulse)
    qcm_sequence = pulses.PulseSequence()
    qcm_sequence.add(qc_pulse)

    tiiq.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    tiiq.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)

    mc = MeasurementControl('MC_T1')
    mc.settables(T1WaitParameter(ro_pulse))
    mc.setpoints(np.arange(delay_before_readout_start,
                           delay_before_readout_end,
                           delay_before_readout_step))
    mc.gettables(Gettable(ROController(tiiq.qrm, tiiq.qcm, qrm_sequence, qcm_sequence)))
    tiiq.LO_qrm.on()
    tiiq.LO_qcm.on()
    tiiq.software_averages = 1 # 3
    dataset = mc.run('T1', soft_avg = tiiq.software_averages)
    tiiq.stop()
    # fit data and determine T1
    # platform.t1 = 

    return dataset


def run_ramsey(resonator_freq, qubit_freq, pi_pulse_gain, pi_pulse_length,
               start_start, start_end, start_step):

    with open("tii_single_qubit_settings.json", "r") as file:
        settings = json.load(file)

    tiiq = TIIq()
    settings['_QCM_settings']['gain'] = pi_pulse_gain
    tiiq.setup(settings) # TODO: Give settings json directory here

    ro_pulse = pulses.TIIReadoutPulse(name="ro_pulse",
                                      start=70,
                                      frequency=20000000.0,
                                      amplitude=0.5,
                                      length=3000,
                                      shape="Block",
                                      delay_before_readout=4)
    qc_pulse = pulses.TIIPulse(name="qc_pulse",
                               start=0,
                               frequency=200000000.0,
                               amplitude=0.3,
                               length=pi_pulse_length//2,
                               shape="Gaussian")
    qc2_pulse = pulses.TIIPulse(name="qc2_pulse",
                               start=pi_pulse_length//2 + 0, # TODO: +0?
                               frequency=200000000.0,
                               amplitude=0.3,
                               length=pi_pulse_length//2,
                               shape="Gaussian")
    qrm_sequence = pulses.PulseSequence()
    qrm_sequence.add(ro_pulse)
    qcm_sequence = pulses.PulseSequence()
    qcm_sequence.add(qc_pulse)
    qcm_sequence.add(qc2_pulse)

    tiiq.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    tiiq.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)

    mc = MeasurementControl('MC_Ramsey')
    mc.settables(RamseyWaitParameter(ro_pulse, qc2_pulse, pi_pulse_length))
    mc.setpoints(np.arange(start_start, start_end, start_step))
    mc.gettables(Gettable(ROController(tiiq.qrm, tiiq.qcm, qrm_sequence, qcm_sequence)))
    tiiq.LO_qrm.on()
    tiiq.LO_qcm.on()
    tiiq.software_averages = 1 # 3
    dataset = mc.run('Ramsey', soft_avg = tiiq.software_averages)
    tiiq.stop()
    # fit data and determine Ramsey Time and dephasing
    # platform.ramsey = 
    # platform.qubit_freq += dephasing
    return dataset


def run_spin_echo(resonator_freq, qubit_freq, pi_pulse_gain, pi_pulse_length,
                  start_start, start_end, start_step):
    with open("tii_single_qubit_settings.json", "r") as file:
        settings = json.load(file)
    tiiq = TIIq()
    # TODO: add set_gain method?
    settings['_QCM_settings']['gain'] = pi_pulse_gain
    tiiq.setup(settings) # TODO: Give settings json directory here

    ro_pulse = pulses.TIIReadoutPulse(name="ro_pulse",
                                      start=70,
                                      frequency=20000000.0,
                                      amplitude=0.5,
                                      length=3000,
                                      shape="Block",
                                      delay_before_readout=4)
    qc_pulse = pulses.TIIPulse(name="qc_pulse",
                               start=0,
                               frequency=200000000.0,
                               amplitude=0.3,
                               length=pi_pulse_length//2,
                               shape="Gaussian")
    qc2_pulse = pulses.TIIPulse(name="qc2_pulse",
                               start=pi_pulse_length//2 + 0, # TODO: +0?
                               frequency=200000000.0,
                               amplitude=0.3,
                               length=pi_pulse_length//2,
                               shape="Gaussian")

    qrm_sequence = pulses.PulseSequence()
    qrm_sequence.add(ro_pulse)
    qcm_sequence = pulses.PulseSequence()
    qcm_sequence.add(qc_pulse)
    qcm_sequence.add(qc2_pulse)

    tiiq.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    tiiq.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)

    mc = MeasurementControl('MC_Spin_Echo')
    mc.settables(SpinEchoWaitParameter(ro_pulse, qc2_pulse, pi_pulse_length))
    mc.setpoints(np.arange(start_start, start_end, start_step))
    mc.gettables(Gettable(ROController(tiiq.qrm, tiiq.qcm, qrm_sequence, qcm_sequence)))
    tiiq.LO_qrm.on()
    tiiq.LO_qcm.on()
    tiiq.software_averages = 1 # 3
    dataset = mc.run('Spin Echo', soft_avg = tiiq.software_averages)
    tiiq.stop()
    
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
        self.qc_pulse.length = value
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
    
    def __init__(self, ro_pulse):
        self.ro_pulse = ro_pulse
        
    def set(self, value):
        # TODO: implement following condition
        #must be >= 4ns <= 65535
        self.ro_pulse.delay_before_readout = value


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
        self.qc2_pulse.start = self.pi_pulse_length//2 + value
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
