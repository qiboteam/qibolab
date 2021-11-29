import numpy as np
from qibolab.platforms import TIIq

from quantify_core.measurement import MeasurementControl
from quantify_core.measurement.control import Settable, Gettable


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

    # Fast Sweep
    tiisq.software_averages = 1
    scanrange = variable_resolution_scanrange(lowres_width= 30e6, lowres_step= 1e6, highres_width= 1e6, highres_step= 0.1e6)

    mc = MeasurementControl('MC')
    mc.settables(tiisq.LO_qrm.frequency)
    mc.setpoints(scanrange + tiisq.LO_qrm.frequency)
    mc.gettables(Gettable(ROController(tiisq.qrm, tiisq.qcm))) # Implement ROController

    tiisq.LO_qrm.on()
    tiisq.LO_qcm.off()

    dataset = mc.run('Resonator Spectroscopy Fast', soft_avg = tiisq._settings['software_averages'])
    # http://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html
    tiisq.stop()
    tiisq.LO_QRM_settings['frequency'] = dataset['x0'].values[dataset['y0'].argmax().values]

    # Precision Sweep
    tiisq._settings['software_averages'] = 1 # 3
    scanrange = np.arange(-0.5e6, 0.5e6, 0.02e6)
    MC.settables(tiisq._LO_qrm.LO.frequency)
    MC.setpoints(scanrange + tiisq._LO_QRM_settings['frequency'])
    MC.gettables(Gettable(ROController(tiisq._qrm, tiisq._qcm)))
    tiisq.LO_qrm.on()
    tiisq.LO_qcm.off()
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
