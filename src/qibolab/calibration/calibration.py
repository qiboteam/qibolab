import pathlib
import numpy as np
#import matplotlib.pyplot as plt
import utils
import yaml
import fitting
from qibolab import Platform

# TODO: Have a look in the documentation of ``MeasurementControl``
#from quantify_core.measurement import MeasurementControl
from quantify_core.measurement.control import Gettable, Settable
from quantify_core.data.handling import set_datadir
from scipy.signal import savgol_filter
from qibolab.pulses import Pulse, ReadoutPulse
from qibolab.circuit import PulseSequence
from qibolab.pulse_shapes import Rectangular, Gaussian


# TODO: Check why this set_datadir is needed
#set_datadir(pathlib.Path("data") / "quantify")
set_datadir(pathlib.Path(__file__).parent / "data" / "quantify")

class Calibration():

    def __init__(self, platform: Platform):
        self.platform = platform
        # TODO: Set mc plotting to false when auto calibrates (default = True for diagnostics)
        self.mc, self.pl, self.ins = utils.create_measurement_control('Calibration', True)

    def load_settings(self):
        # Load diagnostics settings
        with open("calibration.yml", "r") as file:
            return yaml.safe_load(file)

    def run_resonator_spectroscopy(self):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc
        
        ps = platform.settings['settings']
        qc_pulse_shape = eval(ps['qc_spectroscopy_pulse'].popitem()[1])
        qc_pulse_settings = ps['qc_spectroscopy_pulse']
        qc_pulse = Pulse(**qc_pulse_settings, shape = qc_pulse_shape)
        ro_pulse_shape = eval(ps['readout_pulse'].popitem()[1])
        ro_pulse_settings = ps['readout_pulse']
        ro_pulse = ReadoutPulse(**ro_pulse_settings, shape = ro_pulse_shape)
        sequence = PulseSequence()
        sequence.add(qc_pulse)
        sequence.add(ro_pulse)

        ds = self.load_settings()
        self.pl.tuids_max_num(ds['max_num_plots'])
        software_averages = ds['software_averages']
        ds = ds['resonator_spectroscopy']
        lowres_width = ds['lowres_width']
        lowres_step = ds['lowres_step']
        highres_width = ds['highres_width']
        highres_step = ds['highres_step']
        precision_width = ds['precision_width']
        precision_step = ds['precision_step']

        #Fast Sweep
        scanrange = utils.variable_resolution_scanrange(lowres_width, lowres_step, highres_width, highres_step)
        mc.settables(platform.LO_qrm.device.frequency)
        mc.setpoints(scanrange + platform.LO_qrm.get_frequency())
        mc.gettables(Gettable(ROController(platform, sequence)))
        platform.start() 
        platform.LO_qcm.off()
        dataset = mc.run("Resonator Spectroscopy Fast", soft_avg=1)
        platform.stop()
        platform.LO_qrm.set_frequency(dataset['x0'].values[dataset['y0'].argmax().values])
        avg_min_voltage = np.mean(dataset['y0'].values[:(lowres_width//lowres_step)]) * 1e6

        # Precision Sweep
        scanrange = np.arange(-precision_width, precision_width, precision_step)
        mc.settables(platform.LO_qrm.device.frequency)
        mc.setpoints(scanrange + platform.LO_qrm.get_frequency())
        mc.gettables(Gettable(ROController(platform, sequence)))
        platform.start() 
        platform.LO_qcm.off()
        dataset = mc.run("Resonator Spectroscopy Precision", soft_avg=software_averages)
        platform.stop()

        # Fitting
        smooth_dataset = savgol_filter(dataset['y0'].values, 25, 2)
        # resonator_freq = dataset['x0'].values[smooth_dataset.argmax()] + ro_pulse.frequency
        max_ro_voltage = smooth_dataset.max() * 1e6

        f0, BW, Q = fitting.lorentzian_fit("last", max, "Resonator_spectroscopy")
        resonator_freq = (f0*1e9 + ro_pulse.frequency)

        print(f"\nResonator Frequency = {resonator_freq}")
        return resonator_freq, avg_min_voltage, max_ro_voltage, smooth_dataset, dataset

    def run_qubit_spectroscopy(self):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc
        
        ps = platform.settings['settings']
        qc_pulse_shape = eval(ps['qc_spectroscopy_pulse'].popitem()[1])
        qc_pulse_settings = ps['qc_spectroscopy_pulse']
        qc_pulse = Pulse(**qc_pulse_settings, shape = qc_pulse_shape)
        ro_pulse_shape = eval(ps['readout_pulse'].popitem()[1])
        ro_pulse_settings = ps['readout_pulse']
        ro_pulse = ReadoutPulse(**ro_pulse_settings, shape = ro_pulse_shape)
        sequence = PulseSequence()
        sequence.add(qc_pulse)
        sequence.add(ro_pulse)

        ds = self.load_settings()
        self.pl.tuids_max_num(ds['max_num_plots'])
        software_averages = ds['software_averages']
        ds = ds['qubit_spectroscopy']
        fast_start = ds['fast_start']
        fast_end = ds['fast_end']
        fast_step = ds['fast_step']
        precision_start = ds['precision_start']
        precision_end = ds['precision_end']
        precision_step = ds['precision_step']
        
        # Fast Sweep
        fast_sweep_scan_range = np.arange(fast_start, fast_end, fast_step)
        mc.settables(platform.LO_qcm.device.frequency)
        mc.setpoints(fast_sweep_scan_range + platform.LO_qcm.get_frequency())
        mc.gettables(Gettable(ROController(platform, sequence)))
        platform.start() 
        dataset = mc.run("Qubit Spectroscopy Fast", soft_avg=1)
        platform.stop()
        
        # Precision Sweep
        platform.software_averages = 1
        precision_sweep_scan_range = np.arange(precision_start, precision_end, precision_step)
        mc.settables(platform.LO_qcm.device.frequency)
        mc.setpoints(precision_sweep_scan_range + platform.LO_qcm.get_frequency())
        mc.gettables(Gettable(ROController(platform, sequence)))
        platform.start() 
        dataset = mc.run("Qubit Spectroscopy Precision", soft_avg=software_averages)
        platform.stop()

        # Fitting
        smooth_dataset = savgol_filter(dataset['y0'].values, 11, 2)
        qubit_freq = dataset['x0'].values[smooth_dataset.argmin()] - qc_pulse.frequency
        min_ro_voltage = smooth_dataset.min() * 1e6

        print(f"\nQubit Frequency = {qubit_freq}")
        utils.plot(smooth_dataset, dataset, "Qubit_Spectroscopy", 1)
        print("Qubit freq ontained from MC results: ", qubit_freq)
        f0, BW, Q = fitting.lorentzian_fit("last", min, "Qubit_Spectroscopy")
        qubit_freq = (f0*1e9 - qc_pulse.frequency)
        print("Qubit freq ontained from fitting: ", qubit_freq)
        return qubit_freq, min_ro_voltage, smooth_dataset, dataset

    def run_rabi_pulse_length(self):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc
        
        ps = platform.settings['settings']
        qc_pulse_shape = eval(ps['qc_spectroscopy_pulse'].popitem()[1])
        qc_pulse_settings = ps['qc_spectroscopy_pulse']
        qc_pulse = Pulse(**qc_pulse_settings, shape = qc_pulse_shape)
        ro_pulse_shape = eval(ps['readout_pulse'].popitem()[1])
        ro_pulse_settings = ps['readout_pulse']
        ro_pulse = ReadoutPulse(**ro_pulse_settings, shape = ro_pulse_shape)
        sequence = PulseSequence()
        sequence.add(qc_pulse)
        sequence.add(ro_pulse)

        ds = self.load_settings()
        self.pl.tuids_max_num(ds['max_num_plots'])
        software_averages = ds['software_averages']
        ds = ds['rabi_pulse_length']
        pulse_duration_start = ds['pulse_duration_start']
        pulse_duration_end = ds['pulse_duration_end']
        pulse_duration_step = ds['pulse_duration_step']


        mc.settables(Settable(QCPulseLengthParameter(ro_pulse, qc_pulse)))
        mc.setpoints(np.arange(pulse_duration_start, pulse_duration_end, pulse_duration_step))
        mc.gettables(Gettable(ROController(platform, sequence)))
        platform.start()
        dataset = mc.run('Rabi Pulse Length', soft_avg = software_averages)
        platform.stop()
                
        # Fitting
        pi_pulse_amplitude = qc_pulse.amplitude
        smooth_dataset, pi_pulse_duration, rabi_oscillations_pi_pulse_min_voltage, t1 = fitting.rabi_fit(dataset)
        pi_pulse_gain = platform.qcm.gain
        utils.plot(smooth_dataset, dataset, "Rabi_pulse_length", 1)

        print(f"\nPi pulse duration = {pi_pulse_duration}")
        print(f"\nPi pulse amplitude = {pi_pulse_amplitude}") #Check if the returned value from fitting is correct.
        print(f"\nPi pulse gain = {pi_pulse_gain}") #Needed? It is equal to the QCM gain when performing a Rabi.
        print(f"\nrabi oscillation min voltage = {rabi_oscillations_pi_pulse_min_voltage}")
        print(f"\nT1 = {t1}")

        return dataset, pi_pulse_duration, pi_pulse_amplitude, pi_pulse_gain, rabi_oscillations_pi_pulse_min_voltage, t1

    # T1: RX(pi) - wait t(rotates z) - readout
    def run_t1(self):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc
        
        ps = platform.settings['settings']
        start = 0
        frequency = ps['pi_pulse_frequency']
        amplitude = ps['pi_pulse_amplitude']
        duration = ps['pi_pulse_duration']
        phase = 0
        shape = eval(ps['pi_pulse_shape'])
        qc_pi_pulse = Pulse(start, duration, amplitude, frequency, phase, shape)

        ro_pulse_shape = eval(ps['readout_pulse'].popitem()[1])
        ro_pulse_settings = ps['readout_pulse']
        ro_pulse = ReadoutPulse(**ro_pulse_settings, shape = ro_pulse_shape)
        sequence = PulseSequence()
        sequence.add(qc_pi_pulse)
        sequence.add(ro_pulse)

        ds = self.load_settings()
        self.pl.tuids_max_num(ds['max_num_plots'])
        software_averages = ds['software_averages']
        ds = ds['t1']
        delay_before_readout_start = ds['delay_before_readout_start']
        delay_before_readout_end = ds['delay_before_readout_end']
        delay_before_readout_step = ds['delay_before_readout_step']


        mc.settables(Settable(T1WaitParameter(ro_pulse, qc_pi_pulse)))
        mc.setpoints(np.arange(delay_before_readout_start,
                            delay_before_readout_end,
                            delay_before_readout_step))
        mc.gettables(Gettable(ROController(platform, sequence)))
        platform.start()
        dataset = mc.run('T1', soft_avg = software_averages)
        platform.stop()

        # Fitting
        smooth_dataset, t1 = fitting.t1_fit(dataset)
        utils.plot(smooth_dataset, dataset, "t1", 1)
        print(f'\nT1 = {t1}')

        return t1, smooth_dataset, dataset

    def callibrate_qubit_states(self):
   
        platform = self.platform
        platform.reload_settings()
        ps = platform.settings['settings']
        niter=50
        nshots=100

        #create exc and gnd pulses 
        start = 0
        frequency = ps['pi_pulse_frequency']
        amplitude = ps['pi_pulse_amplitude']
        duration = ps['pi_pulse_duration']
        phase = 0
        shape = eval(ps['pi_pulse_shape'])
        qc_pi_pulse = Pulse(start, duration, amplitude, frequency, phase, shape)
    
        #RO pulse starting just after pi pulse
        #ro_pulse_settings = ps['readout_pulse']
        #ro_start = ps['pi_pulse_duration'] + 4 # duration = 11 + 1
        ro_start = 15
        ro_frequency = 20000000
        ro_amplitude = 0.5
        ro_duration = 2000
        ro_phase = 0
        ro_pulse = ReadoutPulse(ro_start, ro_duration, ro_amplitude, ro_frequency, ro_phase, Rectangular())
        
        exc_sequence = PulseSequence()
        exc_sequence.add(qc_pi_pulse)
        exc_sequence.add(ro_pulse)


        gnd_sequence = PulseSequence()
        ro_pulse_shape = eval(ps['readout_pulse'].popitem()[1])
        ro_pulse_settings = ps['readout_pulse']
        ro_pulse = ReadoutPulse(**ro_pulse_settings, shape = ro_pulse_shape)

        gnd_sequence.add(qc_pi_pulse)
        gnd_sequence.add(ro_pulse)

        platform.LO_qrm.set_frequency(ps['resonator_freq'] - ro_pulse.frequency)
        platform.LO_qcm.set_frequency(ps['qubit_freq'] + qc_pi_pulse.frequency)
        platform.start()

        #Exectue niter single exc shots
        all_exc_states = []
        for i in range(niter):
            print(f"Starting exc state calibration {i}")
            qubit_state = platform.execute(exc_sequence, nshots)
            print(f"Finished exc single shot execution  {i}")
            #Compose complex point from i, q obtained from execution
            point = complex(qubit_state[2], qubit_state[3])
            all_exc_states.append(point)

        #Exectue niter single gnd shots
        platform.LO_qcm.off()
        all_gnd_states = []
        for i in range(niter):
            print(f"Starting gnd state calibration  {i}")
            qubit_state = platform.execute(gnd_sequence, nshots)
            print(f"Finished gnd single shot execution  {i}")
            #Compose complex point from i, q obtained from execution
            point = complex(qubit_state[2], qubit_state[3])
            all_gnd_states.append(point)

        platform.stop()

        return all_gnd_states, np.mean(all_gnd_states), all_exc_states, np.mean(all_exc_states)

    def auto_calibrate_plaform(self):
        platform = self.platform

        #backup latest platform runcard
        utils.backup_config_file(platform)

        #run and save cavity spectroscopy calibration
        #resonator_freq, avg_min_voltage, max_ro_voltage, smooth_dataset, dataset = self.run_resonator_spectroscopy()

        # print(utils.get_config_parameter("settings", "", "resonator_freq"))
        # print(utils.get_config_parameter("settings", "", "resonator_spectroscopy_avg_min_ro_voltage"))
        # print(utils.get_config_parameter("settings", "", "resonator_spectroscopy_max_ro_voltage"))
        # print(utils.get_config_parameter("LO_QRM_settings", "", "frequency"))
        # # utils.save_config_parameter("settings", "", "resonator_freq", float(resonator_freq))
        # utils.save_config_parameter("settings", "", "resonator_spectroscopy_avg_min_ro_voltage", float(avg_min_voltage))
        # utils.save_config_parameter("settings", "", "resonator_spectroscopy_max_ro_voltage", float(max_ro_voltage))
        # utils.save_config_parameter("LO_QRM_settings", "", "frequency", float(resonator_freq - 20_000_000))

        #run and save qubit spectroscopy calibration
        # qubit_freq, min_ro_voltage, smooth_dataset, dataset = self.run_qubit_spectroscopy()
        # print(utils.get_config_parameter("settings", "", "qubit_freq"))
        # print(utils.get_config_parameter("LO_QCM_settings", "", "frequency"))
        # print(utils.get_config_parameter("settings", "", "qubit_spectroscopy_min_ro_voltage"))
        # # utils.save_config_parameter("settings", "", "qubit_freq", float(qubit_freq))
        # utils.save_config_parameter("LO_QCM_settings", "", "frequency", float(qubit_freq + 200_000_000))
        # utils.save_config_parameter("settings", "", "qubit_spectroscopy_min_ro_voltage", float(min_ro_voltage))

        # #run Rabi and save Pi pulse params from calibration
        # dataset, pi_pulse_duration, pi_pulse_amplitude, pi_pulse_gain, rabi_oscillations_pi_pulse_min_voltage, t1 = self.run_rabi_pulse_length()
        # print(utils.get_config_parameter("settings", "", "pi_pulse_duration"))
        # print(utils.get_config_parameter("settings", "", "pi_pulse_amplitude"))
        # print(utils.get_config_parameter("settings", "", "pi_pulse_gain"))
        # print(utils.get_config_parameter("settings", "", "rabi_oscillations_pi_pulse_min_voltage"))
        # utils.save_config_parameter("settings", "", "pi_pulse_duration", int(pi_pulse_duration))
        # utils.save_config_parameter("settings", "", "pi_pulse_amplitude", float(pi_pulse_amplitude)) 
        # utils.save_config_parameter("settings", "", "pi_pulse_gain", float(pi_pulse_gain))
        # utils.save_config_parameter("settings", "", "rabi_oscillations_pi_pulse_min_voltage", float(rabi_oscillations_pi_pulse_min_voltage))

        # #run calibration_qubit_states
        all_gnd_states, mean_gnd_states, all_exc_states, mean_exc_states = self.callibrate_qubit_states()
        # #TODO: save in runcard mean_gnd_states and mean_exc_states
        print(all_gnd_states)
        print(mean_gnd_states)
        print(all_exc_states)
        print(mean_exc_states)

        # #TODO: Remove plot qubit states results when tested
        utils.plot_qubit_states(all_gnd_states, all_exc_states)

        #TODO: Remove 0 and 1 classification from auto calibration when tested
        #Classify all points into 0 and 1
        classified_gnd_results = []
        for point in all_gnd_states: 
             classified_gnd_results.append(utils.classify(point, mean_gnd_states, mean_exc_states))

        classified_exc_results = []
        for point in all_exc_states:
             classified_exc_results.append(utils.classify(point, mean_gnd_states, mean_exc_states))

        print(classified_gnd_results)
        print(classified_exc_results)

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