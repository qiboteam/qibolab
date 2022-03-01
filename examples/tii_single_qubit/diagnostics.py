import pathlib
import numpy as np
import matplotlib.pyplot as plt
import yaml

# TODO: Have a look in the documentation of ``MeasurementControl``
from quantify_core.measurement import MeasurementControl
from quantify_core.measurement.control import Gettable, Settable
from quantify_core.data.handling import set_datadir
from scipy.signal import savgol_filter

# TODO: Check why this set_datadir is needed
set_datadir(pathlib.Path(__file__).parent / "data")

def backup_config_file():
    import os
    import shutil
    import errno
    from datetime import datetime
    original = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'qibolab', 'runcards', 'tiiq.yml'))
    now = datetime.now()
    now = now.strftime("%d%m%Y%H%M%S")
    destination_file_name = "tiiq_" + now + ".yml" 
    target = os.path.realpath(os.path.join(os.path.dirname(__file__), 'settings_backups', destination_file_name))

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
    calibration_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'qibolab', 'runcards', 'tiiq.yml'))
    with open(calibration_path) as file:
        settings = yaml.safe_load(file)
    file.close()    

    if (not dictID1):
        return settings[dictID][key]
    else:
        return settings[dictID][dictID1][key]

def save_config_parameter(dictID, dictID1, key, value):
    import os
    calibration_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'qibolab', 'runcards', 'tiiq.yml'))
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
        plt.savefig(label+".pdf")
        return

    if (type == 1): #qubit spec, rabi, ramsey, t1 plots
        fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
        ax.plot(dataset['x0'].values, dataset['y0'].values,'-',color='C0')
        ax.plot(dataset['x0'].values, smooth_dataset,'-',color='C1')
        ax.title.set_text(label)
        ax.plot(dataset['x0'].values[smooth_dataset.argmin()], smooth_dataset[smooth_dataset.argmin()], 'o', color='C2')
        plt.savefig(label+".pdf")
        return

def create_measurement_control(name):
    import os
    if os.environ.get("ENABLE_PLOTMON", True):
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

    def __init__(self, platform, sequence):
        self.platform = platform
        self.sequence = sequence

    def get(self):
        return self.platform.execute(self.sequence)


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

def run_resonator_spectroscopy(platform, mc, sequence, ro_pulse, 
                               lowres_width, lowres_step, highres_width, highres_step,
                               precision_width, precision_step):
    #Fast Sweep
    platform.software_averages = 1
    scanrange = variable_resolution_scanrange(lowres_width, lowres_step, highres_width, highres_step)
    mc.settables(platform.LO_qrm.device.frequency)
    mc.setpoints(scanrange + platform.LO_qrm.get_frequency())
    mc.gettables(Gettable(ROController(platform, sequence)))
    platform.start() 
    #platform.LO_qrm.on()
    platform.LO_qcm.off()
    dataset = mc.run("Resonator Spectroscopy Fast", soft_avg=platform.software_averages)
    platform.stop()
    platform.LO_qrm.set_frequency(dataset['x0'].values[dataset['y0'].argmax().values])
    avg_min_voltage = np.mean(dataset['y0'].values[:25]) * 1e6

    # Precision Sweep
    platform.software_averages = 1
    scanrange = np.arange(-precision_width, precision_width, precision_step)
    mc.settables(platform.LO_qrm.device.frequency)
    mc.setpoints(scanrange + platform.LO_qrm.get_frequency())
    mc.gettables(Gettable(ROController(platform, sequence)))
    platform.start() 
    #platform.LO_qrm.on()
    platform.LO_qcm.off()
    dataset = mc.run("Resonator Spectroscopy Precision", soft_avg=platform.software_averages)
    platform.stop()

    smooth_dataset = savgol_filter(dataset['y0'].values, 25, 2)
    resonator_freq = dataset['x0'].values[smooth_dataset.argmax()] + ro_pulse.frequency
    max_ro_voltage = smooth_dataset.max() * 1e6
    print(f"\nResonator Frequency = {resonator_freq}")
    return resonator_freq, avg_min_voltage, max_ro_voltage, smooth_dataset, dataset

def run_punchout(platform, mc, sequence, ro_pulse, precision_width, precision_step):     
    platform.software_averages = 1
    scanrange = np.arange(-precision_width, precision_width, precision_step)
    scanrange = scanrange + platform.LO_qrm.get_frequency()

    mc.settables([Settable(platform.LO_qrm.device.frequency), Settable(QRPulseGainParameter(platform.qrm))])
    setpoints_gain = np.arange(10, 100, 10)
    mc.setpoints_grid([scanrange, setpoints_gain])
    mc.gettables(Gettable(ROController(platform, sequence)))
    platform.start() 
    #platform.LO_qrm.on()
    platform.LO_qcm.off()
    dataset = mc.run("Punchout", soft_avg=platform.software_averages)
    platform.stop()

    smooth_dataset = savgol_filter(dataset['y0'].values, 25, 2)
    #FIXME: Code Lorentzian fitting for cavity spec and punchout
    resonator_freq = dataset['x0'].values[dataset['y0'].argmax().values]+ro_pulse.frequency 
    print(f"\nResonator Frequency = {resonator_freq}")
    print(f"\nResonator LO Frequency  = {resonator_freq - ro_pulse.frequency}")

    return resonator_freq, smooth_dataset, dataset

def run_qubit_spectroscopy(platform, mc, resonator_freq, sequence, qc_pulse, ro_pulse, 
                           fast_start, fast_end, fast_step,
                           precision_start, precision_end, precision_step):
    
    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    platform.software_averages = 1
    # Fast Sweep
    fast_sweep_scan_range = np.arange(fast_start, fast_end, fast_step)
    mc.settables(platform.LO_qcm.device.frequency)
    mc.setpoints(fast_sweep_scan_range + platform.LO_qcm.get_frequency())
    mc.gettables(Gettable(ROController(platform, sequence)))
    platform.start() 
    dataset = mc.run("Qubit Spectroscopy Fast", soft_avg=platform.software_averages)
    platform.stop()
    
    # Precision Sweep
    platform.software_averages = 1
    precision_sweep_scan_range = np.arange(precision_start, precision_end, precision_step)
    mc.settables(platform.LO_qcm.device.frequency)
    mc.setpoints(precision_sweep_scan_range + platform.LO_qcm.get_frequency())
    mc.gettables(Gettable(ROController(platform, sequence)))
    platform.start() 
    dataset = mc.run("Qubit Spectroscopy Precision", soft_avg=platform.software_averages)
    platform.stop()

    smooth_dataset = savgol_filter(dataset['y0'].values, 11, 2)
    qubit_freq = dataset['x0'].values[smooth_dataset.argmin()] - qc_pulse.frequency
    min_ro_voltage = smooth_dataset.min() * 1e6

    return qubit_freq, min_ro_voltage, smooth_dataset, dataset

def run_rabi_pulse_length(platform, mc, resonator_freq, qubit_freq, sequence, qc_pulse, ro_pulse):

    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    platform.LO_qcm.set_frequency(qubit_freq - qc_pulse.frequency)
    platform.software_averages = 3
    mc.settables(Settable(QCPulseLengthParameter(ro_pulse, qc_pulse)))
    mc.setpoints(np.arange(1, 400, 1))
    mc.gettables(Gettable(ROController(platform, sequence)))
    platform.start()
    dataset = mc.run('Rabi Pulse Length', soft_avg = platform.software_averages)
    platform.stop()
    
    return dataset, platform.qcm.gain

def run_rabi_pulse_gain(platform, mc, resonator_freq, qubit_freq, sequence, qc_pulse, ro_pulse):

    #qubit pulse duration=200
    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    platform.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)
    platform.software_averages = 3
    mc.settables(Settable(QCPulseGainParameter(platform.qcm)))
    mc.setpoints(np.arange(0, 100, 10))
    mc.gettables(Gettable(ROController(platform, sequence)))
    platform.start()
    dataset = mc.run('Rabi Pulse Gain', soft_avg = platform.software_averages)
    platform.stop()
    
    return dataset

def run_rabi_pulse_length_and_gain(platform, mc, resonator_freq, qubit_freq, sequence, qc_pulse, ro_pulse):

    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    platform.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)
    platform.software_averages = 1
    mc.settables([Settable(QCPulseLengthParameter(ro_pulse, qc_pulse)),
                  Settable(QCPulseGainParameter(platform.qcm))])
    setpoints_length = np.arange(1, 400, 10)
    setpoints_gain = np.arange(0, 20, 1)
    mc.setpoints_grid([setpoints_length, setpoints_gain])
    mc.gettables(Gettable(ROController(platform, sequence)))
    platform.start()
    dataset = mc.run('Rabi Pulse Length and Gain', soft_avg = platform.software_averages)
    # Analyse data to look for the smallest qc_pulse length that renders off-resonance amplitude, determine corresponding pi_pulse gain
    # platform.pi_pulse_length =
    # platform.pi_pulse_gain =
    platform.stop()
    
    return dataset

def run_rabi_pulse_length_and_amplitude(platform, mc, resonator_freq, qubit_freq, sequence, qc_pulse, ro_pulse):

    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    platform.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)
    platform.software_averages = 1
    mc.settables([Settable(QCPulseLengthParameter(ro_pulse, qc_pulse)),
                  Settable(QCPulseAmplitudeParameter(qc_pulse))])
    setpoints_length = np.arange(1, 1000, 2)
    setpoints_amplitude = np.arange(0, 100, 2)
    mc.setpoints_grid([setpoints_length, setpoints_amplitude])
    mc.gettables(Gettable(ROController(platform, sequence)))
    platform.start()
    dataset = mc.run('Rabi Pulse Length and Gain', soft_avg = platform.software_averages)
    # Analyse data to look for the smallest qc_pulse length that renders off-resonance amplitude, determine corresponding pi_pulse gain
    # platform.pi_pulse_length =
    # platform.pi_pulse_gain =
    platform.stop()
    
    return dataset

def run_t1(platform, mc,resonator_freq, qubit_freq, sequence, qc_pulse, ro_pulse,                
           pi_pulse_gain, pi_pulse_duration, delay_before_readout_start, 
           delay_before_readout_end, delay_before_readout_step):

    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    platform.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)
    platform.qcm.gain = pi_pulse_gain

    mc.settables(Settable(T1WaitParameter(ro_pulse, qc_pulse)))
    mc.setpoints(np.arange(delay_before_readout_start,
                           delay_before_readout_end,
                           delay_before_readout_step))
    mc.gettables(Gettable(ROController(platform, sequence)))
    platform.start()
    dataset = mc.run('T1', soft_avg = platform.software_averages)
    platform.stop()

    return dataset

def run_ramsey(platform, mc, resonator_freq, qubit_freq, sequence, qc_pulse, qc2_pulse, ro_pulse, 
               pi_pulse_gain, pi_pulse_duration, start_start, start_end, start_step):

    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    platform.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)
    platform.qcm.gain = pi_pulse_gain

    mc.settables(Settable(RamseyWaitParameter(ro_pulse, qc2_pulse, pi_pulse_duration)))
    mc.setpoints(np.arange(start_start, start_end, start_step))
    mc.gettables(Gettable(ROController(platform, sequence)))
    platform.start()
    dataset = mc.run('Ramsey', soft_avg = platform.software_averages)
    platform.stop()

    return dataset

def run_spin_echo(platform, mc, resonator_freq, qubit_freq, sequence, qc_pulse, qc2_pulse, ro_pulse,
                  pi_pulse_gain, pi_pulse_length, pi_pulse_amplitude,
                  start_start, start_end, start_step):

    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    platform.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)
    platform.qcm.gain = pi_pulse_gain
    mc.settables(Settable(SpinEchoWaitParameter(ro_pulse, qc2_pulse, pi_pulse_length)))
    mc.setpoints(np.arange(start_start, start_end, start_step))
    mc.gettables(Gettable(ROController(platform, sequence)))
    platform.start()
    dataset = mc.run('Spin Echo', soft_avg = platform.software_averages)
    platform.stop()
    
    return dataset

# Spin Echo 3 Pulses: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - RX(pi/2) - readout
def run_spin_echo_3pulses(platform, mc, resonator_freq, qubit_freq, sequence, qc_pulse, qc2_pulse, qc3_pulse, ro_pulse,
                          pi_pulse_gain, pi_pulse_length, pi_pulse_amplitude,
                          start_start, start_end, start_step):
    
    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    platform.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)
    platform.qcm.gain = pi_pulse_gain
    mc.settables(SpinEcho3PWaitParameter(ro_pulse, qc2_pulse, qc3_pulse, pi_pulse_length))
    mc.setpoints(np.arange(start_start, start_end, start_step))
    mc.gettables(Gettable(ROController(platform, sequence)))
    platform.start()
    dataset = mc.run('Spin Echo 3 Pulses', soft_avg = platform.software_averages)
    platform.stop()

    return dataset

def run_shifted_resonator_spectroscopy(platform, mc, resonator_freq, qubit_freq, sequence, qc_pulse, ro_pulse,
                                       lowres_width, lowres_step, highres_width, highres_step,
                                       precision_width, precision_step):

    platform.LO_qrm.set_frequency(resonator_freq - ro_pulse.frequency)
    platform.LO_qcm.set_frequency(qubit_freq + qc_pulse.frequency)

    # Fast Sweep
    platform.software_averages = 1
    scanrange = variable_resolution_scanrange(lowres_width, lowres_step, highres_width, highres_step)
    mc.settables(platform.LO_qrm.device.frequency)
    mc.setpoints(scanrange + platform.LO_qrm.get_frequency())
    mc.gettables(Gettable(ROController(platform, sequence)))
    platform.start()
    dataset = mc.run("Resonator Spectroscopy Shifted Fast", soft_avg=platform.software_averages)
    platform.stop()

    shifted_LO_frequency = dataset['x0'].values[dataset['y0'].argmax().values]

    # Precision Sweep
    platform.software_averages = 1
    scanrange = np.arange(-precision_width, precision_width, precision_step)
    mc.settables(platform.LO_qrm.device.frequency)
    mc.setpoints(scanrange + shifted_LO_frequency)
    mc.gettables(Gettable(ROController(platform, sequence)))
    platform.start()
    dataset = mc.run("Resonator Spectroscopy Shifted Precision", soft_avg=platform.software_averages)
    platform.stop()

    smooth_dataset = savgol_filter(dataset['y0'].values, 25, 2)
    shifted_frequency = dataset['x0'].values[smooth_dataset.argmax()] + ro_pulse.frequency
    shifted_max_ro_voltage = smooth_dataset.max() * 1e6
    print('\n')
    print(f"\nResonator Frequency = {shifted_frequency}")
    print(f"Maximum Voltage Measured = {shifted_max_ro_voltage} μV")

    return shifted_frequency, shifted_max_ro_voltage, smooth_dataset, dataset



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

class SpinEcho3PWaitParameter():
    label = 'Time'
    unit = 'ns'
    name = 'spin_echo_wait'
    initial_value = 0
    
    def __init__(self, ro_pulse, qc2_pulse, qc3_pulse, pi_pulse_length):
        self.ro_pulse = ro_pulse
        self.qc2_pulse = qc2_pulse
        self.qc3_pulse = qc3_pulse
        self.pi_pulse_length = pi_pulse_length
        
    def set(self,value):
        self.qc2_pulse.start = self.pi_pulse_length//2 + value
        self.qc3_pulse.start = (3 * self.pi_pulse_length)//2 + 2 * value
        self.ro_pulse.start = 2 * self.pi_pulse_length + 2 * value + 4

class QRPulseGainParameter():

    label = 'Qubit Readout Gain'
    unit = '%'
    name = 'ro_pulse_gain'

    def __init__(self, qrm):
        self.qrm = qrm

    def set(self,value):
        self.qrm.gain = value / 100