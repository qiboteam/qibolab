import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from qibolab import pulses, platform
from qibolab.pulse_shapes import Rectangular, Gaussian

# TODO: Have a look in the documentation of ``MeasurementControl``
from quantify_core.measurement import MeasurementControl
from quantify_core.measurement.control import Gettable, Settable
from quantify_core.data.handling import set_datadir
# TODO: Check why this set_datadir is needed
set_datadir(pathlib.Path(__file__).parent / "data")


def create_measurement_control(name):
    import os
    if os.environ.get("ENABLE_PLOTMON", False):
        mc = MeasurementControl(f'MC {name}')
        from quantify_core.visualization.pyqt_plotmon import PlotMonitor_pyqt
        plotmon = PlotMonitor_pyqt(f'Plot Monitor {name}')
        plotmon.tuids_max_num(3)
        mc.instr_plotmon(plotmon.name)
        from quantify_core.visualization.instrument_monitor import InstrumentMonitor
        insmon = InstrumentMonitor(f"Instruments Monitor {name}")
        mc.instrument_monitor(insmon.name)
        return mc, plotmon, insmon
    else:
        mc = MeasurementControl(f'MC {name}')
        return mc, None, None


class ROController():
    # Quantify Gettable Interface Implementation
    label = ['Amplitude', 'Phase','I','Q']
    unit = ['V', 'Radians','V','V']
    name = ['A', 'Phi','I','Q']

    def __init__(self, sequence):
        self.sequence = sequence

    def get(self):
        return platform(self.sequence)


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


def get_pulse_sequence(duration=4000):
    qc_pulse = pulses.Pulse(start=0,
                            frequency=200000000.0,
                            amplitude=0.9,
                            duration=duration,
                            phase=0,
                            shape=Gaussian(4000 / 5))
    ro_pulse = pulses.ReadoutPulse(start=duration + 4,
                                   frequency=20000000.0,
                                   amplitude=0.9,
                                   duration=2000,
                                   phase=0,
                                   shape=Rectangular())
    sequence = pulses.PulseSequence()
    sequence.add(qc_pulse)
    sequence.add(ro_pulse)
    return sequence, qc_pulse, ro_pulse


def run_resonator_spectroscopy(lowres_width, lowres_step,
                               highres_width, highres_step,
                               precision_width, precision_step):
    sequence, qc_pulse, ro_pulse = get_pulse_sequence()

    mc, pl, ins = create_measurement_control('resonator_spectroscopy')
    # Fast Sweep
    platform.software_averages = 1
    scanrange = variable_resolution_scanrange(lowres_width, lowres_step, highres_width, highres_step)
    mc.settables(platform.LO_qrm.device.frequency)
    mc.setpoints(scanrange + platform.LO_qrm.get_frequency())
    mc.gettables(Gettable(ROController(sequence)))
    platform.LO_qrm.on()
    platform.LO_qcm.off()
    dataset = mc.run("Resonator Spectroscopy Fast", soft_avg=platform.software_averages)
    # http://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html
    platform.stop()
    platform.LO_qrm.set_frequency(dataset['x0'].values[dataset['y0'].argmax().values])

    # Precision Sweep
    platform.software_averages = 1 # 3
    scanrange = np.arange(-precision_width, precision_width, precision_step)
    mc.settables(platform.LO_qrm.device.frequency)
    mc.setpoints(scanrange + platform.LO_qrm.get_frequency())
    mc.gettables(Gettable(ROController(sequence)))
    platform.LO_qrm.on()
    platform.LO_qcm.off()
    dataset = mc.run("Resonator Spectroscopy Precision", soft_avg=platform.software_averages)
    platform.stop()

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


def run_qubit_spectroscopy(resonator_freq, fast_start, fast_end, fast_step,
                           precision_start, precision_end, precision_step):
    sequence, qc_pulse, ro_pulse = get_pulse_sequence()

    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    mc, pl, ins = create_measurement_control('qubit_spectroscopy')
    # Fast Sweep
    platform.software_averages = 1
    scanrange = np.arange(fast_start, fast_end, fast_step)
    mc.settables(platform.LO_qcm.device.frequency)
    mc.setpoints(scanrange + platform.LO_qcm.get_frequency())
    mc.gettables(Gettable(ROController(sequence)))
    platform.LO_qrm.on()
    platform.LO_qcm.on()
    dataset = mc.run("Qubit Spectroscopy Fast", soft_avg=platform.software_averages)
    platform.stop()
    # http://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html
    platform.LO_qcm.set_frequency(dataset['x0'].values[dataset['y0'].argmin().values])

    # Precision Sweep
    platform.software_averages = 3
    scanrange = np.arange(precision_start, precision_end, precision_step)
    mc.settables(platform.LO_qcm.device.frequency)
    mc.setpoints(scanrange + platform.LO_qcm.get_frequency())
    mc.gettables(Gettable(ROController(sequence)))
    platform.LO_qrm.on()
    platform.LO_qcm.on()
    dataset = mc.run("Qubit Spectroscopy Precision", soft_avg=platform.software_averages)
    platform.stop()

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
    sequence, qc_pulse, ro_pulse = get_pulse_sequence()

    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    platform.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)
    mc, pl, ins = create_measurement_control('Rabi_pulse_length')
    platform.software_averages = 1
    mc.settables(Settable(QCPulseLengthParameter(ro_pulse, qc_pulse)))
    mc.setpoints(np.arange(1, 200, 1))
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
    setpoints_length = np.arange(1, 400, 2)
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

    mc = MeasurementControl('MC_T1')
    mc.settables(Settable(T1WaitParameter(ro_pulse)))
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
                               length=pi_pulse_length // 2,
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

    mc = MeasurementControl('MC_Ramsey')
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
                             length=pi_pulse_length // 2,
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

    mc = MeasurementControl('MC_Spin_Echo')
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
