import argparse
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


parser = argparse.ArgumentParser()
parser.add_argument("--lowres-width", default=30e6, type=float)
parser.add_argument("--lowres-step", default=1e6, type=float)
parser.add_argument("--highres-width", default=1e6, type=float)
parser.add_argument("--highres-step", default=0.1e6, type=float)
parser.add_argument("--precision-width", default=0.5e6, type=float)
parser.add_argument("--precision-step", default=0.02e6, type=float)


class ROController():
    # TODO: ``ROController`` implementation
    # This should be the complicated part as it involves the pulses

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
        # TODO: Find a better way to pass the frequency of readout pulse here
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
                               length=60,
                               shape="Gaussian")
    qrm_sequence = pulses.PulseSequence()
    qrm_sequence.add(ro_pulse)
    qcm_sequence = pulses.PulseSequence()
    qcm_sequence.add(qc_pulse)

    mc = MeasurementControl('MC')
    # Fast Sweep
    tiiq.software_averages = 1
    # TODO: Make the following arguments of the main function and add argument parser
    scanrange = variable_resolution_scanrange(lowres_width, lowres_step, highres_width, highres_step)
    mc.settables(tiiq.LO_qrm.frequency)
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
    mc.settables(tiiq.LO_qrm.frequency)
    mc.setpoints(scanrange + tiiq.LO_qrm.get_frequency())
    mc.gettables(Gettable(ROController(tiiq.qrm, tiiq.qcm, qrm_sequence, qcm_sequence)))
    tiiq.LO_qrm.on()
    tiiq.LO_qcm.off()
    dataset = mc.run("Resonator Spectroscopy Precision", soft_avg=tiiq.software_averages)
    tiiq.stop()

    # TODO: Add ``savgol_filter`` method
    from scipy.signal import savgol_filter
    smooth_dataset = savgol_filter(dataset['y0'].values, 25, 2)
    # TODO: is the following call really needed given that the oscillator is never used after that?
    tiiq.LO_qrm.set_frequency(dataset['x0'].values[smooth_dataset.argmax()])

    # TODO: Remove ``_QRM_settings`` from here given that we will use a different pulse mechanism
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
    return dataset


if __name__ == "__main__":
    args = vars(parser.parse_args())
    run_resonator_spectroscopy(**args)
