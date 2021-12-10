import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from qibolab import pulses
from qibolab.pulse_shapes import Rectangular, Gaussian
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

    def __init__(self, qrm, qcm, sequence):
        self.qrm = qrm
        self.qcm = qcm
        self.sequence = sequence

    def get(self):
        #self.qcm.setup(qcm._settings)
        waveforms, program = self.qcm.translate(self.sequence)
        self.qcm.upload(waveforms, program, "./data")

        #self.qrm.setup(qrm._settings) # this has already been done earlier?
        waveforms, program = self.qrm.translate(self.sequence)
        self.qrm.upload(waveforms, program, "./data")

        self.qcm.play_sequence()
        acquisition_results = self.qrm.play_sequence_and_acquire(self.sequence.readout_pulses[0])
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
    ro_pulse = pulses.ReadoutPulse(start=70,
                                   frequency=20000000.0,
                                   amplitude=0.5,
                                   duration=3000,
                                   phase=0,
                                   shape=Rectangular(),
                                   delay_before_readout=4)
    qc_pulse = pulses.Pulse(start=0,
                            frequency=200000000.0,
                            amplitude=0.3,
                            duration=60,
                            phase=0,
                            shape=Gaussian(60 / 5))
    sequence = pulses.PulseSequence()
    sequence.add(qc_pulse)
    sequence.add(ro_pulse)

    tiiq = TIIq()
    mc = MeasurementControl('MC_resonator_spectroscopy')
    # Fast Sweep
    tiiq.software_averages = 1
    scanrange = variable_resolution_scanrange(lowres_width, lowres_step, highres_width, highres_step)
    mc.settables(tiiq.LO_qrm.device.frequency)
    mc.setpoints(scanrange + tiiq.LO_qrm.get_frequency())
    mc.gettables(Gettable(ROController(tiiq.qrm, tiiq.qcm, sequence)))
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
    mc.gettables(Gettable(ROController(tiiq.qrm, tiiq.qcm, sequence)))
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


def run_qubit_spectroscopy(resonator_freq, fast_start, fast_end, fast_step,
                           precision_start, precision_end, precision_step):
    ro_pulse = pulses.ReadoutPulse(start=70,
                                   frequency=20000000.0,
                                   amplitude=0.5,
                                   duration=4040,
                                   phase=0,
                                   shape=Rectangular(),
                                   delay_before_readout=4)
    qc_pulse = pulses.Pulse(start=0,
                            frequency=200000000.0,
                            amplitude=0.3,
                            duration=4000,
                            phase=0,
                            shape=Gaussian(4000 / 5))
    sequence = pulses.PulseSequence()
    sequence.add(qc_pulse)
    sequence.add(ro_pulse)

    tiiq = TIIq()
    # set optimal resonator frequency found in ``run_resonator_spectroscopy``
    tiiq.LO_qrm.set_frequency(resonator_freq)

    mc = MeasurementControl('MC_qubit_spectroscopy')
    # Fast Sweep
    tiiq.software_averages = 1
    scanrange = np.arange(fast_start, fast_end, fast_step)
    mc.settables(tiiq.LO_qcm.device.frequency)
    mc.setpoints(scanrange + tiiq.LO_qcm.get_frequency())
    mc.gettables(Gettable(ROController(tiiq.qrm, tiiq.qcm, sequence)))
    tiiq.LO_qrm.on()
    tiiq.LO_qcm.on()
    dataset = mc.run("Qubit Spectroscopy Fast", soft_avg=tiiq.software_averages)
    tiiq.stop()
    # http://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html
    tiiq.LO_qcm.set_frequency(dataset['x0'].values[dataset['y0'].argmin().values])

    # Precision Sweep
    tiiq.software_averages = 3
    scanrange = np.arange(precision_start, precision_end, precision_step)
    mc.settables(tiiq.LO_qcm.device.frequency)
    mc.setpoints(scanrange + tiiq.LO_qcm.get_frequency())
    mc.gettables(Gettable(ROController(tiiq.qrm, tiiq.qcm, sequence)))
    tiiq.LO_qrm.on()
    tiiq.LO_qcm.on()
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
