import pathlib
from qibolab.paths import qibolab_folder
from quantify_core.measurement import MeasurementControl
import numpy as np
import matplotlib.pyplot as plt
import pathlib

script_folder = pathlib.Path(__file__).parent



data_folder = qibolab_folder / "calibration" / "data"
data_folder.mkdir(parents=True, exist_ok=True)

quantify_folder = qibolab_folder / "calibration" / "data" / "quantify"
quantify_folder.mkdir(parents=True, exist_ok=True)

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

def plot(smooth_dataset, dataset, label, type):
    if (type == 0): #cavity plots
        fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
        ax.plot(dataset['x0'].values, dataset['y0'].values,'-',color='C0')
        ax.plot(dataset['x0'].values, smooth_dataset,'-',color='C1')
        ax.title.set_text(label)
        ax.plot(dataset['x0'].values[smooth_dataset.argmax()], smooth_dataset[smooth_dataset.argmax()], 'o', color='C2')
        plt.savefig(pathlib.Path("data") / f"{label}.pdf")
        return

    if (type == 1): #qubit spec, rabi, ramsey, t1 plots
        fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
        ax.plot(dataset['x0'].values, dataset['y0'].values,'-',color='C0')
        ax.plot(dataset['x0'].values, smooth_dataset,'-',color='C1')
        ax.title.set_text(label)
        ax.plot(dataset['x0'].values[smooth_dataset.argmin()], smooth_dataset[smooth_dataset.argmin()], 'o', color='C2')
        plt.savefig(data_folder / f"{label}.pdf")
        return

def plot_qubit_states(gnd_results, exc_results):
    plt.figure(figsize=[4,4])
    # Plot all the results
    # All results from the gnd_schedule are plotted in blue
    plt.scatter(np.real(gnd_results), np.imag(gnd_results), s=5, cmap='viridis', c='blue', alpha=0.5, label='state_0')
    # All results from the exc_schedule are plotted in red
    plt.scatter(np.real(exc_results), np.imag(exc_results), s=5, cmap='viridis', c='red', alpha=0.5, label='state_1')

    # Plot a large dot for the average result of the 0 and 1 states.
    mean_gnd = np.mean(gnd_results) # takes mean of both real and imaginary parts
    mean_exc = np.mean(exc_results)
    plt.scatter(np.real(mean_gnd), np.imag(mean_gnd), s=200, cmap='viridis', c='black',alpha=1.0, label='state_0_mean')
    plt.scatter(np.real(mean_exc), np.imag(mean_exc), s=200, cmap='viridis', c='black',alpha=1.0, label='state_1_mean')

    plt.ylabel('I [a.u.]', fontsize=15)
    plt.xlabel('Q [a.u.]', fontsize=15)
    plt.title("0-1 discrimination", fontsize=15)
    #plt.show()
    plt.savefig( data_folder / "qubit_states_classification.pdf")


def create_measurement_control(name, debug=True):
    import os
    from qibo.config import log
    log.info(f"Creating MeasurementControl {name}")
    if os.environ.get("ENABLE_PLOTMON", debug):
        mc = MeasurementControl(f'MC_{name}')
        from quantify_core.visualization.pyqt_plotmon import PlotMonitor_pyqt
        plotmon = PlotMonitor_pyqt(f'Plot_Monitor_{name}')
        mc.instr_plotmon(plotmon.name)
        from quantify_core.visualization.instrument_monitor import InstrumentMonitor
        insmon = InstrumentMonitor(f"Instruments_Monitor_{name}")
        return mc, plotmon, insmon
    else:
        mc = MeasurementControl(f'MC_{name}')
        return mc, None, None
    # TODO: be able to choose which windows are opened and remember their sizes and dimensions


def classify(point: complex, mean_gnd, mean_exc):
        import math
        """Classify the given state as |0> or |1>."""
        def distance(a, b):
            return math.sqrt((np.real(a) - np.real(b))**2 + (np.imag(a) - np.imag(b))**2)

        return int(distance(point, mean_exc) < distance(point, mean_gnd))

    
      