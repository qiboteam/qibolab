from quantify_core.measurement import MeasurementControl
import numpy as np
import yaml
import matplotlib.pyplot as plt
import pathlib

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

def check_data_dir():
    import os

    # You should change 'test' to your preferred folder.
    MYDIR = ("data")
    CHECK_FOLDER = os.path.isdir(MYDIR)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)


def backup_config_file(platform):
    import os
    import shutil
    import errno
    from datetime import datetime
    original = str(platform.runcard)
    now = datetime.now()
    now = now.strftime("%d%m%Y%H%M%S")
    destination_file_name = "tiiq_" + now + ".yml"
    target = os.path.realpath(os.path.join(os.path.dirname(__file__), 'data/settings_backups', destination_file_name))

    try:
        print("Copying file: " + original)
        print("Destination file" + target)
        shutil.copyfile(original, target)
        print("Platform settings backup done")
    except IOError as e:
        # ENOENT(2): file does not exist, raised also on missing dest parent dir
        if e.errno != errno.ENOENT:
            raise
            # try creating parent directories
        os.makedirs(os.path.dirname(target))
        shutil.copy(original, target)

def get_config_parameter(dictID, dictID1, key):
    import os
    calibration_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'runcards', 'tiiq.yml'))
    with open(calibration_path) as file:
        settings = yaml.safe_load(file)
    file.close()

    if (not dictID1):
        return settings[dictID][key]
    else:
        return settings[dictID][dictID1][key]

def save_config_parameter(dictID, dictID1, key, value):
    import os
    calibration_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'runcards', 'tiiq.yml'))
    with open(calibration_path, "r") as file:
        settings = yaml.safe_load(file)
    file.close()

    if (not dictID1):
        settings[dictID][key] = value
        print("Saved value: " + str(settings[dictID][key]))

    else:
        settings[dictID][dictID1][key] = value
        print("Saved value: " + str(settings[dictID][dictID1][key]))

    with open(calibration_path, "w") as file:
        settings = yaml.dump(settings, file, sort_keys=False, indent=4)
    file.close()

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
        plt.savefig(pathlib.Path("data") / f"{label}.pdf")
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
    plt.savefig(pathlib.Path("data") / "qubit_states_classification.pdf")


def create_measurement_control(name, debug=True):
    import os
    if os.environ.get("ENABLE_PLOTMON", debug):
        mc = MeasurementControl(f'MC {name}')
        from quantify_core.visualization.pyqt_plotmon import PlotMonitor_pyqt
        plotmon = PlotMonitor_pyqt(f'Plot Monitor {name}')
        mc.instr_plotmon(plotmon.name)
        from quantify_core.visualization.instrument_monitor import InstrumentMonitor
        insmon = InstrumentMonitor(f"Instruments Monitor {name}")
        mc.instrument_monitor(insmon.name)
        return mc, plotmon, insmon
    else:
        mc = MeasurementControl(f'MC {name}')
        return mc, None, None
    # TODO: be able to choose which windows are opened and remember their sizes and dimensions


def classify(point: complex, mean_gnd, mean_exc):
        import math
        """Classify the given state as |0> or |1>."""
        def distance(a, b):
            return math.sqrt((np.real(a) - np.real(b))**2 + (np.imag(a) - np.imag(b))**2)

        return int(distance(point, mean_exc) < distance(point, mean_gnd))
