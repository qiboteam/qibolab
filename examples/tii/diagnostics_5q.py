# Characterisation can be done by changing settings to qibolab/runcards/tiiq.yml and diagnostics.yml
# These scripts do not save the characterisation results on the runcard; to do so use 
#   ds.backup_config_file()
#   resonator_freq, avg_min_voltage, max_ro_voltage, smooth_dataset, dataset = ds.run_resonator_spectroscopy()
#   ds.save_config_parameter('resonator_freq', resonator_freq, 'characterization', 'single_qubit', qubit)

from qibolab.paths import qibolab_folder
from qibolab import Platform
from qibolab.calibration import utils

import pathlib
from scipy.signal import savgol_filter
from qibolab.paths import qibolab_folder
import numpy as np
import matplotlib.pyplot as plt
import yaml

from qibolab import Platform
from qibolab.paths import qibolab_folder
from qibolab.calibration import utils
from qibolab.calibration import fitting
from qibolab.pulses import Pulse, ReadoutPulse, Rectangular, Gaussian, Drag
from qibolab.circuit import PulseSequence


class Diagnostics():

    def __init__(self, platform: Platform, settings_file = None,  show_plots=True):
        self.platform = platform
        if not settings_file:
            script_folder = pathlib.Path(__file__).parent
            settings_file = script_folder / "diagnostics.yml"
        self.settings_file = settings_file
        # TODO: Set mc plotting to false when auto calibrates (default = True for diagnostics)
        self.mc, self.pl, self.ins = utils.create_measurement_control('Calibration', show_plots)
        self.mcs = {}

    def load_settings(self):
        # Load calibration settings
        with open(self.settings_file, "r") as file:
            self.settings = yaml.safe_load(file)
            self.software_averages = self.settings['software_averages']
            self.software_averages_precision = self.settings['software_averages_precision']
            self.max_num_plots = self.settings['max_num_plots']

    def reload_settings(self):
        self.load_settings()

    #--------------------------#
    # Single qubit experiments #
    #--------------------------#

    def run_resonator_spectroscopy(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        sequence = PulseSequence()
        ro_pulse = platform.qubit_readout_pulse(qubit, start = 0)
        sequence.add(ro_pulse)

        self.reload_settings()
        self.lowres_width = self.settings['resonator_spectroscopy']['lowres_width']
        self.lowres_step = self.settings['resonator_spectroscopy']['lowres_step']
        self.highres_width = self.settings['resonator_spectroscopy']['highres_width']
        self.highres_step = self.settings['resonator_spectroscopy']['highres_step']
        self.precision_width = self.settings['resonator_spectroscopy']['precision_width']
        self.precision_step = self.settings['resonator_spectroscopy']['precision_step']

        self.pl.tuids_max_num(self.max_num_plots)
        platform.qrm[qubit].lo.frequency = platform.characterization['single_qubit'][qubit]['resonator_freq'] - ro_pulse.frequency
        #Fast Sweep
        if (self.software_averages !=0):
            scanrange = utils.variable_resolution_scanrange(self.lowres_width, self.lowres_step, self.highres_width, self.highres_step)
            mc.settables(SettableFrequency(platform.qrm[qubit].lo))
            mc.setpoints(scanrange + platform.qrm[qubit].lo.frequency)
            mc.gettables(ROController(platform, sequence, qubit))
            platform.start() 
            dataset = mc.run("Resonator Spectroscopy Fast", soft_avg=self.software_averages)
            platform.stop()
            platform.qrm[qubit].lo.frequency = dataset['x0'].values[dataset['y0'].argmin().values]
            avg_max_voltage = np.mean(dataset['y0'].values[:(self.lowres_width//self.lowres_step)]) * 1e6

        # Precision Sweep
        if (self.software_averages_precision !=0):
            scanrange = np.arange(-self.precision_width, self.precision_width, self.precision_step)
            mc.settables(SettableFrequency(platform.qrm[qubit].lo))
            mc.setpoints(scanrange + platform.qrm[qubit].lo.frequency)
            mc.gettables(ROController(platform, sequence, qubit))
            platform.start()
            dataset = mc.run("Resonator Spectroscopy Precision", soft_avg=self.software_averages_precision)
            platform.stop()

        # Fitting
        smooth_dataset = savgol_filter(dataset['y0'].values, 25, 2)
        # resonator_freq = dataset['x0'].values[smooth_dataset.argmax()] + ro_pulse.frequency
        min_ro_voltage = smooth_dataset.min() * 1e6

        f0, BW, Q = fitting.lorentzian_fit("last", min, "Resonator_spectroscopy")
        resonator_freq = (f0*1e9 + ro_pulse.frequency)

        print(f"\nResonator Frequency = {resonator_freq}")
        return resonator_freq, avg_max_voltage, min_ro_voltage, smooth_dataset, dataset
        
    




    def run_resonator_spectroscopy_flux(self, qubit=0, fluxline=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc


        sequence = PulseSequence()
        ro_pulse = platform.qubit_readout_pulse(qubit, start = 0)
        sequence.add(ro_pulse)

        self.reload_settings()
        self.lowres_width = self.settings['resonator_spectroscopy']['lowres_width']
        self.lowres_step = self.settings['resonator_spectroscopy']['lowres_step']
        self.highres_width = self.settings['resonator_spectroscopy']['highres_width']
        self.highres_step = self.settings['resonator_spectroscopy']['highres_step']
        self.precision_width = self.settings['resonator_spectroscopy']['precision_width']
        self.precision_step = self.settings['resonator_spectroscopy']['precision_step']

        self.pl.tuids_max_num(self.max_num_plots)

        spi = platform.instruments['SPI'].device
        spi.set_dacs_zero()

        # freqs = [platform.characterization['single_qubit'][qubit]['resonator_freq'] - ro_pulse.frequency for qubit in range(6)]
        freq = platform.characterization['single_qubit'][qubit]['resonator_freq'] - ro_pulse.frequency
        around = 5e6
        freqs = np.linspace(freq-around, freq+around, 300)
        dacs = [spi.mod2.dac0, spi.mod1.dac0, spi.mod1.dac1, spi.mod1.dac2, spi.mod1.dac3]
        flux = np.linspace(-30e-3, 30e-3, 40)

        mc.setpoints_grid([freqs, flux])
        mc.settables([SettableFrequency(platform.qrm[qubit].lo), dacs[fluxline].current])
        mc.gettables(ROController(platform, sequence, qubit))
        platform.start() 
        data = mc.run(name="matrix3")
        platform.stop()
        spi.set_dacs_zero()

    
    
    
    def run_qubit_spectroscopy(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        sequence = PulseSequence()
        qd_pulse = platform.qubit_drive_pulse(qubit, start = 0, duration = 5000) 
        ro_pulse = platform.qubit_readout_pulse(qubit, start = 5000)
        sequence.add(qd_pulse)
        sequence.add(ro_pulse)

        self.reload_settings()
        self.fast_start = self.settings['qubit_spectroscopy']['fast_start']
        self.fast_end = self.settings['qubit_spectroscopy']['fast_end']
        self.fast_step = self.settings['qubit_spectroscopy']['fast_step']
        self.precision_start = self.settings['qubit_spectroscopy']['precision_start']
        self.precision_end = self.settings['qubit_spectroscopy']['precision_end']
        self.precision_step = self.settings['qubit_spectroscopy']['precision_step']

        self.pl.tuids_max_num(self.max_num_plots)
        platform.qrm[qubit].lo.frequency = platform.characterization['single_qubit'][qubit]['resonator_freq'] - ro_pulse.frequency
        
        # Fast Sweep
        if (self.software_averages !=0):
            platform.qcm[qubit].lo.frequency = platform.characterization['single_qubit'][qubit]['qubit_freq'] + qd_pulse.frequency
            lo_qcm_frequency = platform.qcm[qubit].lo.frequency
            fast_sweep_scan_range = np.arange(self.fast_start, self.fast_end, self.fast_step)
            mc.settables(SettableFrequency(platform.qcm[qubit].lo))
            mc.setpoints(fast_sweep_scan_range + lo_qcm_frequency)
            mc.gettables(ROController(platform, sequence, qubit))
            platform.start() 
            dataset = mc.run("Qubit Spectroscopy Fast", soft_avg=self.software_averages)
            platform.stop()
            platform.qcm[qubit].lo.frequency = dataset['x0'].values[dataset['y0'].argmax().values]
            avg_min_voltage = np.mean(dataset['y0'].values[:((self.fast_end - self.fast_start)//self.lowres_step)]) * 1e6

        # Precision Sweep
        if (self.software_averages_precision !=0):
            lo_qcm_frequency = platform.qcm[qubit].lo.frequency
            precision_sweep_scan_range = np.arange(self.precision_start, self.precision_end, self.precision_step)
            mc.settables(SettableFrequency(platform.qcm[qubit].lo))
            mc.setpoints(precision_sweep_scan_range + lo_qcm_frequency)
            mc.gettables(ROController(platform, sequence, qubit))
            platform.start() 
            dataset = mc.run("Qubit Spectroscopy Precision", soft_avg=self.software_averages_precision)
            platform.stop()

        # Fitting
        smooth_dataset = savgol_filter(dataset['y0'].values, 11, 2)
        qubit_freq = dataset['x0'].values[smooth_dataset.argmax()] - qd_pulse.frequency
        max_ro_voltage = smooth_dataset.max() * 1e6

        print(f"\nQubit Frequency = {qubit_freq}")
        utils.plot(smooth_dataset, dataset, "Qubit_Spectroscopy", 1)
        print("Qubit freq ontained from MC results: ", qubit_freq)
        f0, BW, Q = fitting.lorentzian_fit("last", max, "Qubit_Spectroscopy")
        qubit_freq = (f0*1e9 - qd_pulse.frequency)
        print("Qubit freq ontained from fitting: ", qubit_freq)
        return qubit_freq, max_ro_voltage, smooth_dataset, dataset
    
    def run_rabi_pulse_length(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        sequence = PulseSequence()
        qd_pulse = platform.qubit_drive_pulse(qubit, start = 0, duration = 4) 
        ro_pulse = platform.qubit_readout_pulse(qubit, start = 4)
        sequence.add(qd_pulse)
        sequence.add(ro_pulse)

        self.reload_settings()
        self.pulse_duration_start = self.settings['rabi_pulse_length']['pulse_duration_start']
        self.pulse_duration_end = self.settings['rabi_pulse_length']['pulse_duration_end']
        self.pulse_duration_step = self.settings['rabi_pulse_length']['pulse_duration_step']

        self.pl.tuids_max_num(self.max_num_plots)
        platform.qrm[qubit].lo.frequency = platform.characterization['single_qubit'][qubit]['resonator_freq'] - ro_pulse.frequency
        platform.qcm[qubit].lo.frequency = platform.characterization['single_qubit'][qubit]['qubit_freq'] + qd_pulse.frequency

        mc.settables(QCPulseLengthParameter(ro_pulse, qd_pulse))
        mc.setpoints(np.arange(self.pulse_duration_start, self.pulse_duration_end, self.pulse_duration_step))
        mc.gettables(ROController(platform, sequence, qubit))
        platform.start()
        dataset = mc.run('Rabi Pulse Length', soft_avg = self.software_averages)
        platform.stop()

        # Fitting
        pi_pulse_amplitude = qd_pulse.amplitude
        smooth_dataset, pi_pulse_duration, rabi_oscillations_pi_pulse_min_voltage, t1 = fitting.rabi_fit(dataset)
        utils.plot(smooth_dataset, dataset, "Rabi_pulse_length", 1)

        print(f"\nPi pulse duration = {pi_pulse_duration}")
        print(f"\nPi pulse amplitude = {pi_pulse_amplitude}") #Check if the returned value from fitting is correct.
        print(f"\nrabi oscillation min voltage = {rabi_oscillations_pi_pulse_min_voltage}")
        print(f"\nT1 = {t1}")

        return dataset, pi_pulse_duration, pi_pulse_amplitude, rabi_oscillations_pi_pulse_min_voltage, t1


    # T1: RX(pi) - wait t(rotates z) - readout
    def run_t1(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc
        
        sequence = PulseSequence()
        RX_pulse = platform.RX_pulse(qubit, start = 0)
        ro_pulse = platform.qubit_readout_pulse(qubit, start = RX_pulse.duration)
        sequence.add(RX_pulse)
        sequence.add(ro_pulse)

        self.reload_settings()
        self.delay_before_readout_start = self.settings['t1']['delay_before_readout_start']
        self.delay_before_readout_end = self.settings['t1']['delay_before_readout_end']
        self.delay_before_readout_step = self.settings['t1']['delay_before_readout_step']

        self.pl.tuids_max_num(self.max_num_plots)
        platform.qrm[qubit].lo.frequency = platform.characterization['single_qubit'][qubit]['resonator_freq'] - ro_pulse.frequency
        platform.qcm[qubit].lo.frequency = platform.characterization['single_qubit'][qubit]['qubit_freq'] + RX_pulse.frequency
        
        mc.settables(T1WaitParameter(ro_pulse, RX_pulse))
        mc.setpoints(np.arange( self.delay_before_readout_start,
                                self.delay_before_readout_end,
                                self.delay_before_readout_step))
        mc.gettables(ROController(platform, sequence, qubit))
        platform.start()
        dataset = mc.run('T1', soft_avg = self.software_averages)
        platform.stop()

        # Fitting
        smooth_dataset, t1 = fitting.t1_fit(dataset)
        utils.plot(smooth_dataset, dataset, "t1", 1)
        print(f'\nT1 = {t1}')

        return t1, smooth_dataset, dataset


    # Ramsey: RX(pi/2) - wait t(rotates z) - RX(pi/2) - readout
    def run_ramsey(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        sequence = PulseSequence()
        RX90_pulse1 = platform.RX90_pulse(qubit, start = 0)
        RX90_pulse2 = platform.RX90_pulse(qubit, start = RX90_pulse1.duration)
        ro_pulse = platform.qubit_readout_pulse(qubit, start = RX90_pulse1.duration + RX90_pulse2.duration)
        sequence.add(RX90_pulse1)
        sequence.add(RX90_pulse2)
        sequence.add(ro_pulse)
        
        self.reload_settings()
        self.delay_between_pulses_start = self.settings['ramsey']['delay_between_pulses_start']
        self.delay_between_pulses_end = self.settings['ramsey']['delay_between_pulses_end']
        self.delay_between_pulses_step = self.settings['ramsey']['delay_between_pulses_step']

        self.pl.tuids_max_num(self.max_num_plots)
        platform.qrm[qubit].lo.frequency = platform.characterization['single_qubit'][qubit]['resonator_freq'] - ro_pulse.frequency
        platform.qcm[qubit].lo.frequency = platform.characterization['single_qubit'][qubit]['qubit_freq'] + RX90_pulse1.frequency

        mc.settables(RamseyWaitParameter(ro_pulse, RX90_pulse2))
        mc.setpoints(np.arange(self.delay_between_pulses_start, self.delay_between_pulses_end, self.delay_between_pulses_step))
        mc.gettables(ROController(platform, sequence, qubit))
        platform.start()
        dataset = mc.run('Ramsey', soft_avg = self.software_averages)
        platform.stop()

        # Fitting
        smooth_dataset, delta_frequency, t2 = fitting.ramsey_fit(dataset)
        utils.plot(smooth_dataset, dataset, "Ramsey", 1)
        print(f"\nDelta Frequency = {delta_frequency}")
        print(f"\nT2 = {t2} ns")

        return delta_frequency, t2, smooth_dataset, dataset
















class SettableFrequency():
        label = 'Frequency'
        unit = 'Hz'
        name = 'frequency'
        
        def __init__(self, instance):
            self.instance = instance

        def set(self, value):
            self.instance.frequency =  value

class QCPulseLengthParameter():

    label = 'Qubit Control Pulse Length'
    unit = 'ns'
    name = 'qd_pulse_length'

    def __init__(self, ro_pulse, qd_pulse):
        self.ro_pulse = ro_pulse
        self.qd_pulse = qd_pulse

    def set(self, value):
        self.qd_pulse.duration = value
        self.ro_pulse.start = value

class T1WaitParameter():
    label = 'Time'
    unit = 'ns'
    name = 't1_wait'
    initial_value = 0

    def __init__(self, ro_pulse, qd_pulse):
        self.ro_pulse = ro_pulse
        self.qd_pulse = qd_pulse

    def set(self, value):
        # TODO: implement following condition
        #must be >= 4ns <= 65535
        #platform.delay_before_readout = value
        self.ro_pulse.start = self.qd_pulse.duration + value

class RamseyWaitParameter():
    label = 'Time'
    unit = 'ns'
    name = 'ramsey_wait'
    initial_value = 0

    def __init__(self, ro_pulse, qc2_pulse):
        self.ro_pulse = ro_pulse
        self.qc2_pulse = qc2_pulse
        self.pulse_length = qc2_pulse.duration

    def set(self, value):
        self.qc2_pulse.start = self.pulse_length  + value
        self.ro_pulse.start = self.pulse_length * 2 + value 
        
class ROController():
    # Quantify Gettable Interface Implementation
    label = ['Amplitude', 'Phase','I','Q']
    unit = ['V', 'Radians','V','V']
    name = ['A', 'Phi','I','Q']

    def __init__(self, platform, sequence, qubit):
        self.platform = platform
        self.sequence = sequence
        self.qubit = qubit

    def get(self):
        results = self.platform.execute_pulse_sequence(self.sequence)
        return list(list(results.values())[0].values())[0] #TODO: Replace with the particular acquisition





if __name__=='__main__':
    import os
    script_folder = pathlib.Path(os.path.abspath(''))
    diagnostics_settings = script_folder / 'examples' / 'tii' / "diagnostics.yml"

    runcard = qibolab_folder / 'runcards' / 'qw5q.yml' 

    # Create a platform; connect and configure it
    platform = Platform('multiqubit', runcard)
    platform.connect()
    platform.setup()

    # create a diagnostics/calibration object
    ds = Diagnostics(platform, diagnostics_settings)

    qubit = 0

    resonator_freq, avg_max_voltage, min_ro_voltage, smooth_dataset, dataset = ds.run_resonator_spectroscopy(qubit)
    qubit_freq, max_ro_voltage, smooth_dataset, dataset = ds.run_qubit_spectroscopy(qubit)

    for qubit in range(5):
        resonator_freq, avg_max_voltage, min_ro_voltage, smooth_dataset, dataset = ds.run_resonator_spectroscopy(qubit)

    for qubit in range(5):
        qubit_freq, max_ro_voltage, smooth_dataset, dataset = ds.run_qubit_spectroscopy(qubit)



