import numpy as np
from qibolab import pulses
from qibolab.platforms import TIIq


# TODO: Have a look in the documentation of ``MeasurementControl``
from quantify_core.measurement import MeasurementControl
from quantify_core.measurement.control import Gettable


class ROController():
    # TODO: ``ROController`` implementation
    # This should be the complicated part as it involves the pulses

    # Quantify Gettable Interface Implementation
    label = ['Amplitude', 'Phase','I','Q']
    unit = ['V', 'Radians','V','V']
    name = ['A', 'Phi','I','Q']

    def __init__(self, qrm, qcm, qrm_pulses, qcm_pulses):
        self._qrm = qrm
        self._qcm = qcm
        self.qrm_pulses = qrm_pulses
        self.qcm_pulses = qcm_pulses

    def get(self):
        qrm = self._qrm
        qcm = self._qcm

        #qrm.setup(qrm._settings) # this has already been done earlier?
        waveform = qrm.translate(self.qrm_pulses)
        qrm.set_program_from_parameters(qrm._settings)
        qrm.set_acquisitions()
        qrm.set_weights()
        qrm.upload_sequence()

        #qcm.setup(qcm._settings)
        qcm.set_waveforms_from_pulses_definition(qcm._settings['pulses'])
        qcm.set_program_from_parameters(qcm._settings)
        qcm.set_acquisitions()
        qcm.set_weights()
        qcm.upload_sequence()

        qcm.play_sequence()
        acquisition_results = qrm.play_sequence_and_acquire()
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


def run_resonator_spectroscopy():
    tiiq = TIIq()
    tiiq.setup() # TODO: Give settings json directory here

    mc = MeasurementControl('MC')
    qrm_pulses = [pulses.TIIPulse(
                            name="ro_pulse",
                            frequency=20000000.0,
                            amplitude=0.9,
                            #start=340,
                            length=6000,
                            shape="Block")]
    qcm_pulses = [pulses.TIIPulse(
                            name="qc_pulse",
                            frequency=200e6,
                            amplitude=0.25,
                            length=300,
                            #delay_before=10,
                            shape="Gaussian")]

    # Fast Sweep
    tiiq.software_averages = 1
    # TODO: Make the following arguments of the main function and add argument parser
    scanrange = variable_resolution_scanrange(lowres_width= 30e6, lowres_step= 1e6, highres_width= 1e6, highres_step= 0.1e6)
    mc.settables(tiiq.LO_qrm.frequency)
    mc.setpoints(scanrange + tiiq.LO_qrm.frequency)
    mc.gettables(Gettable(ROController(tiiq.qrm, tiiq.qcm, qrm_pulses, qcm_pulses)))

    tiiq.LO_qrm.on()
    tiiq.LO_qcm.off()

    dataset = mc.run("Resonator Spectroscopy Fast", soft_avg=tiiq.software_averages)
    # http://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html
    tiiq.stop()
    tiiq.LO_qrm.set_frequency(dataset['x0'].values[dataset['y0'].argmax().values])

    # Precision Sweep
    tiiq.software_averages = 1 # 3
    scanrange = np.arange(-0.5e6, 0.5e6, 0.02e6)
    mc.settables(tiiq.LO_qrm.frequency)
    mc.setpoints(scanrange + tiiq.LO_qrm.frequency)
    mc.gettables(Gettable(ROController(tiiq.qrm, tiiq.qcm)))
    tiiq.LO_qrm.on()
    tiiq.LO_qcm.off()
    dataset = MC.run("Resonator Spectroscopy Precision", soft_avg=tiiq.software_averages)
    tiiq.stop()

    # TODO: Add ``savgol_filter`` method
    smooth_dataset = savgol_filter(dataset['y0'].values, 25, 2)
    # TODO: is the following call really needed given that the oscillator is never used after that?
    tiiq.LO_qrm.set_frequency(dataset['x0'].values[smooth_dataset.argmax()])

    # TODO: Remove ``_QRM_settings`` from here given that we will use a different pulse mechanism
    resonator_freq = dataset['x0'].values[smooth_dataset.argmax()] + tiiq._QRM_settings['pulses']['ro_pulse']['freq_if']
    print(f"Resonator Frequency = {resonator_freq}")
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
