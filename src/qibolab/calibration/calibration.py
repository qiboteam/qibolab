import pathlib
from scipy.signal import savgol_filter
from qibolab.paths import qibolab_folder
import numpy as np
import matplotlib.pyplot as plt
import yaml

from quantify_core.measurement import MeasurementControl
from quantify_core.measurement.control import Gettable, Settable
from quantify_core.data.handling import set_datadir

from qibolab import Platform
from qibolab.paths import qibolab_folder
from qibolab.calibration import utils
from qibolab.calibration import fitting
from qibolab.pulses import Pulse, ReadoutPulse, Rectangular, Gaussian, Drag
from qibolab.circuit import PulseSequence


class Calibration():

    def __init__(self, platform: Platform, settings_file = None,  show_plots=True):
        self.platform = platform
        if not settings_file:
            script_folder = pathlib.Path(__file__).parent
            settings_file = script_folder / "calibration.yml"
        self.settings_file = settings_file
        # TODO: Set mc plotting to false when auto calibrates (default = True for diagnostics)
        self.mc, self.pl, self.ins = utils.create_measurement_control('Calibration', show_plots)

    def load_settings(self):
        # Load calibration settings
        with open(self.settings_file, "r") as file:
            self.settings = yaml.safe_load(file)
        self.__dict__.update(self.settings)
        return self.settings

    def reload_settings(self):
        self.load_settings()

    #--------------------------#
    # Single qubit experiments #
    #--------------------------#

    def run_resonator_spectroscopy(self, qubit=1):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc
        ro_channel = platform.ro_channel[qubit]
        qrm = platform.qrm[qubit]
        lo_qrm = platform.lo_qrm[qubit]

        qd_channel = platform.qd_channel[qubit]
        qcm = platform.qcm[qubit]
        lo_qcm = platform.lo_qcm[qubit]


        ps = platform.settings['settings']
        ro_pulse_settings = ps['readout_pulse']
        ro_pulse = ReadoutPulse(**ro_pulse_settings, channel = ro_channel)
        sequence = PulseSequence()
        sequence.add(ro_pulse)

        self.reload_settings()
        self.__dict__.update(self.settings['resonator_spectroscopy'])
        self.pl.tuids_max_num(self.max_num_plots)

        #Fast Sweep
        if (self.software_averages !=0):
            scanrange = utils.variable_resolution_scanrange(self.lowres_width, self.lowres_step, self.highres_width, self.highres_step)
            mc.settables(lo_qrm.device.frequency)
            mc.setpoints(scanrange + lo_qrm.frequency)
            mc.gettables(Gettable(ROController(platform, sequence)))
            platform.start() 
            lo_qcm.off()
            dataset = mc.run("Resonator Spectroscopy Fast", soft_avg=self.software_averages)
            platform.stop()
            lo_qrm.frequency = (dataset['x0'].values[dataset['y0'].argmax().values])
            avg_min_voltage = np.mean(dataset['y0'].values[:(self.lowres_width//self.lowres_step)]) * 1e6

        # Precision Sweep
        if (self.software_averages_precision !=0):
            scanrange = np.arange(-self.precision_width, self.precision_width, self.precision_step)
            mc.settables(lo_qrm.device.frequency)
            mc.setpoints(scanrange + lo_qrm.frequency)
            mc.gettables(Gettable(ROController(platform, sequence)))
            platform.start() 
            lo_qcm.off()
            dataset = mc.run("Resonator Spectroscopy Precision", soft_avg=self.software_averages_precision)
            platform.stop()

        # Fitting
        smooth_dataset = savgol_filter(dataset['y0'].values, 25, 2)
        # resonator_freq = dataset['x0'].values[smooth_dataset.argmax()] + ro_pulse.frequency
        max_ro_voltage = smooth_dataset.max() * 1e6

        f0, BW, Q = fitting.lorentzian_fit("last", max, "Resonator_spectroscopy")
        resonator_freq = (f0*1e9 + ro_pulse.frequency)

        print(f"\nResonator Frequency = {resonator_freq}")
        return resonator_freq, avg_min_voltage, max_ro_voltage, smooth_dataset, dataset

    def run_qubit_spectroscopy(self, qubit=1):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc
        ro_channel = platform.ro_channel[qubit]
        qrm = platform.qrm[qubit]
        lo_qrm = platform.lo_qrm[qubit]

        qd_channel = platform.qd_channel[qubit]
        qcm = platform.qcm[qubit]
        lo_qcm = platform.lo_qcm[qubit]

        ps = platform.settings['settings']
        qd_pulse_settings = ps['qd_spectroscopy_pulse']
        qd_pulse = Pulse(**qd_pulse_settings, channel = qd_channel)
        ro_pulse_settings = ps['readout_pulse']
        ro_pulse = ReadoutPulse(**ro_pulse_settings, channel = ro_channel)
        sequence = PulseSequence()
        sequence.add(qd_pulse)
        sequence.add(ro_pulse)

        self.reload_settings()
        self.__dict__.update(self.settings['qubit_spectroscopy'])
        self.pl.tuids_max_num(self.max_num_plots)
        
        # Fast Sweep
        if (self.software_averages !=0):
            lo_qcm_frequency = lo_qcm.frequency
            fast_sweep_scan_range = np.arange(self.fast_start, self.fast_end, self.fast_step)
            mc.settables(lo_qcm.device.frequency)
            mc.setpoints(fast_sweep_scan_range + lo_qcm.frequency)
            mc.gettables(Gettable(ROController(platform, sequence)))
            platform.start() 
            dataset = mc.run("Qubit Spectroscopy Fast", soft_avg=self.software_averages)
            platform.stop()

        # Precision Sweep
        if (self.software_averages_precision !=0):
            lo_qcm.frequency = lo_qcm_frequency
            precision_sweep_scan_range = np.arange(self.precision_start, self.precision_end, self.precision_step)
            mc.settables(lo_qcm.device.frequency)
            mc.setpoints(precision_sweep_scan_range + lo_qcm.frequency)
            mc.gettables(Gettable(ROController(platform, sequence)))
            platform.start() 
            dataset = mc.run("Qubit Spectroscopy Precision", soft_avg=self.software_averages_precision)
            platform.stop()

        # Fitting
        smooth_dataset = savgol_filter(dataset['y0'].values, 11, 2)
        qubit_freq = dataset['x0'].values[smooth_dataset.argmin()] - qd_pulse.frequency
        min_ro_voltage = smooth_dataset.min() * 1e6

        print(f"\nQubit Frequency = {qubit_freq}")
        utils.plot(smooth_dataset, dataset, "Qubit_Spectroscopy", 1)
        print("Qubit freq ontained from MC results: ", qubit_freq)
        f0, BW, Q = fitting.lorentzian_fit("last", min, "Qubit_Spectroscopy")
        qubit_freq = (f0*1e9 - qd_pulse.frequency)
        print("Qubit freq ontained from fitting: ", qubit_freq)
        return qubit_freq, min_ro_voltage, smooth_dataset, dataset

    def run_rabi_pulse_length(self, qubit=1):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc
        ro_channel = platform.ro_channel[qubit]
        qrm = platform.qrm[qubit]
        lo_qrm = platform.lo_qrm[qubit]

        qd_channel = platform.qd_channel[qubit]
        qcm = platform.qcm[qubit]
        lo_qcm = platform.lo_qcm[qubit]

        ps = platform.settings['settings']
        qd_pulse_settings = ps['qd_spectroscopy_pulse']
        qd_pulse = Pulse(**qd_pulse_settings, channel = qd_channel)
        ro_pulse_settings = ps['readout_pulse']
        ro_pulse = ReadoutPulse(**ro_pulse_settings, channel = ro_channel)
        sequence = PulseSequence()
        sequence.add(qd_pulse)
        sequence.add(ro_pulse)

        self.reload_settings()
        self.__dict__.update(self.settings['rabi_pulse_length'])
        self.pl.tuids_max_num(self.max_num_plots)

        mc.settables(Settable(QCPulseLengthParameter(ro_pulse, qd_pulse)))
        mc.setpoints(np.arange(self.pulse_duration_start, self.pulse_duration_end, self.pulse_duration_step))
        mc.gettables(Gettable(ROController(platform, sequence)))
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
    def run_t1(self, qubit=1):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc
        qd_channel = platform.settings['qubit_channel_map'][1][1]
        ro_channel = platform.settings['qubit_channel_map'][1][0]
        
        RX_pulse = platform.settings['native_gates']['single_qubit'][qubit]['RX']
        qc_pi_pulse = Pulse(**RX_pulse, channel = qd_channel)
        ps = platform.settings['settings']
        ro_pulse_settings = ps['readout_pulse']
        ro_pulse = ReadoutPulse(**ro_pulse_settings, channel = ro_channel)
        sequence = PulseSequence()
        sequence.add(qc_pi_pulse)
        sequence.add(ro_pulse)

        self.reload_settings()
        self.__dict__.update(self.settings['t1'])
        self.pl.tuids_max_num(self.max_num_plots)

        mc.settables(Settable(T1WaitParameter(ro_pulse, qc_pi_pulse)))
        mc.setpoints(np.arange( self.delay_before_readout_start,
                                self.delay_before_readout_end,
                                self.delay_before_readout_step))
        mc.gettables(Gettable(ROController(platform, sequence)))
        platform.start()
        dataset = mc.run('T1', soft_avg = self.software_averages)
        platform.stop()

        # Fitting
        smooth_dataset, t1 = fitting.t1_fit(dataset)
        utils.plot(smooth_dataset, dataset, "t1", 1)
        print(f'\nT1 = {t1}')

        return t1, smooth_dataset, dataset

    # Ramsey: RX(pi/2) - wait t(rotates z) - RX(pi/2) - readout
    def run_ramsey(self, qubit=1):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc
        qd_channel = platform.settings['qubit_channel_map'][1][1]
        ro_channel = platform.settings['qubit_channel_map'][1][0]
        
        RX_pulse = platform.settings['native_gates']['single_qubit'][qubit]['RX']
        RX90_pulse = RX_pulse.copy()
        RX90_pulse.update({'amplitude': RX_pulse['amplitude']/2})
        
        qc_pi_half_pulse_1 = Pulse(**RX90_pulse, channel = qd_channel)
        qc_pi_half_pulse_2 = Pulse(**RX90_pulse, channel = qd_channel)
        qc_pi_half_pulse_2.start = qc_pi_half_pulse_1.start + qc_pi_half_pulse_1.duration
        
        ps = platform.settings['settings']
        ro_pulse_settings = ps['readout_pulse']
        ro_pulse = ReadoutPulse(**ro_pulse_settings, channel = ro_channel)
        sequence = PulseSequence()
        sequence.add(qc_pi_half_pulse_1)
        sequence.add(qc_pi_half_pulse_2)
        sequence.add(ro_pulse)
        
        self.reload_settings()
        self.__dict__.update(self.settings['ramsey'])
        self.pl.tuids_max_num(self.max_num_plots)

        mc.settables(Settable(RamseyWaitParameter(ro_pulse, qc_pi_half_pulse_2)))
        mc.setpoints(np.arange(self.delay_between_pulses_start, self.delay_between_pulses_end, self.delay_between_pulses_step))
        mc.gettables(Gettable(ROController(platform, sequence)))
        platform.start()
        dataset = mc.run('Ramsey', soft_avg = self.software_averages)
        platform.stop()

        # Fitting
        smooth_dataset, delta_frequency, t2 = fitting.ramsey_fit(dataset)
        utils.plot(smooth_dataset, dataset, "Ramsey", 1)
        print(f"\nDelta Frequency = {delta_frequency}")
        print(f"\nT2 = {t2} ns")

        return delta_frequency, t2, smooth_dataset, dataset

    def calibrate_qubit_states(self):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc
        qd_channel = platform.settings['qubit_channel_map'][1][1]
        ro_channel = platform.settings['qubit_channel_map'][1][0]

        ps = platform.settings['settings']
        niter=100
        nshots=1

        #create exc and gnd pulses 
        start = 0
        frequency = ps['pi_pulse_frequency']
        amplitude = ps['pi_pulse_amplitude']
        duration = ps['pi_pulse_duration']
        phase = 0
        shape =ps['pi_pulse_shape']
        qc_pi_pulse = Pulse(start, duration, amplitude, frequency, phase, shape)
    
        #RO pulse starting just after pi pulse
        ro_start = ps['pi_pulse_duration'] + 4
        ro_pulse_settings = ps['readout_pulse']
        ro_duration = ro_pulse_settings['duration']
        ro_amplitude = ro_pulse_settings['amplitude']
        ro_frequency = ro_pulse_settings['frequency']
        ro_phase = ro_pulse_settings['phase']
        ro_pulse = ReadoutPulse(ro_start, ro_duration, ro_amplitude, ro_frequency, ro_phase, Rectangular())
        
        exc_sequence = PulseSequence()
        exc_sequence.add(qc_pi_pulse)
        exc_sequence.add(ro_pulse)

        platform.start()
        #Exectue niter single exc shots
        all_exc_states = []
        for i in range(niter):
            print(f"Starting exc state calibration {i}")
            qubit_state = platform.execute_pulse_sequence(exc_sequence, 1) # TODO: Improve the speed of this with binning
            qubit_state = list(list(qubit_state.values())[0].values())[0]
            print(f"Finished exc single shot execution  {i}")
            #Compose complex point from i, q obtained from execution
            point = complex(qubit_state[2], qubit_state[3])
            all_exc_states.append(point)
        platform.stop()

        ro_pulse_shape =ps['readout_pulse']['shape']
        ro_pulse_settings = ps['readout_pulse']
        ro_pulse = ReadoutPulse(**ro_pulse_settings, channel = ro_channel, shape = ro_pulse_shape)

        gnd_sequence = PulseSequence()
        gnd_sequence.add(qc_pi_pulse)
        gnd_sequence.add(ro_pulse)

        platform.lo_qrm.frequency = (ps['resonator_freq'] - ro_pulse.frequency)
        platform.lo_qcm.frequency = (ps['qubit_freq'] + qc_pi_pulse.frequency)
        #Exectue niter single gnd shots
        platform.start()
        platform.lo_qcm.off()
        all_gnd_states = []
        for i in range(niter):
            print(f"Starting gnd state calibration  {i}")
            qubit_state = platform.execute_pulse_sequence(gnd_sequence, 1) # TODO: Improve the speed of this with binning
            qubit_state = list(list(qubit_state.values())[0].values())[0]
            print(f"Finished gnd single shot execution  {i}")
            #Compose complex point from i, q obtained from execution
            point = complex(qubit_state[2], qubit_state[3])
            all_gnd_states.append(point)
        platform.stop()

        return all_gnd_states, np.mean(all_gnd_states), all_exc_states, np.mean(all_exc_states)
   
    def auto_calibrate_plaform(self):
        platform = self.platform

        #backup latest platform runcard
        self.backup_config_file()
        for qubit in platform.qubits:
            # run and save cavity spectroscopy calibration
            resonator_freq, avg_min_voltage, max_ro_voltage, smooth_dataset, dataset = self.run_resonator_spectroscopy(qubit)
            self.save_config_parameter("resonator_freq", int(resonator_freq), 'characterization', 'single_qubit', qubit)
            self.save_config_parameter("resonator_spectroscopy_avg_min_ro_voltage", float(avg_min_voltage), 'characterization', 'single_qubit', qubit)
            self.save_config_parameter("resonator_spectroscopy_max_ro_voltage", float(max_ro_voltage), 'characterization', 'single_qubit', qubit)
            lo_qrm_frequency = int(resonator_freq - platform.native_gates['single_qubit'][qubit]['MZ']['frequency'])
            self.save_config_parameter("frequency", lo_qrm_frequency, 'instruments', platform.lo_qrm[qubit].name, 'settings') # TODO: cambiar IF hardcoded

            # run and save qubit spectroscopy calibration
            qubit_freq, min_ro_voltage, smooth_dataset, dataset = self.run_qubit_spectroscopy(qubit)
            self.save_config_parameter("qubit_freq", int(qubit_freq), 'characterization', 'single_qubit', qubit)
            RX_pulse_sequence = platform.native_gates['single_qubit'][qubit]['RX']['pulse_sequence']
            lo_qcm_frequency = int(qubit_freq + RX_pulse_sequence[0]['frequency'])
            self.save_config_parameter("frequency", lo_qcm_frequency, 'instruments', platform.lo_qcm[qubit].name, 'settings')
            self.save_config_parameter("qubit_spectroscopy_min_ro_voltage", float(min_ro_voltage), 'characterization', 'single_qubit', qubit)

            # run Rabi and save Pi pulse calibration
            dataset, pi_pulse_duration, pi_pulse_amplitude, rabi_oscillations_pi_pulse_min_voltage, t1 = self.run_rabi_pulse_length(qubit)
            RX_pulse_sequence[0]['duration'] = int(pi_pulse_duration)
            RX_pulse_sequence[0]['amplitude'] = float(pi_pulse_amplitude)
            self.save_config_parameter("pulse_sequence", RX_pulse_sequence, 'native_gates', 'single_qubit', qubit, 'RX')
            self.save_config_parameter("rabi_oscillations_pi_pulse_min_voltage", float(rabi_oscillations_pi_pulse_min_voltage), 'characterization', 'single_qubit', qubit)

            # run Ramsey and save T2 calibration
            delta_frequency, t2, smooth_dataset, dataset = self.run_ramsey(qubit)
            adjusted_qubit_freq = int(platform.characterization['single_qubit'][qubit]['qubit_freq'] + delta_frequency)
            self.save_config_parameter("qubit_freq", adjusted_qubit_freq, 'characterization', 'single_qubit', qubit)
            self.save_config_parameter("T2", float(t2), 'characterization', 'single_qubit', qubit)
            RX_pulse_sequence = platform.native_gates['single_qubit'][qubit]['RX']['pulse_sequence']
            lo_qcm_frequency = int(adjusted_qubit_freq + RX_pulse_sequence[0]['frequency'])
            self.save_config_parameter("frequency", lo_qcm_frequency, 'instruments', platform.lo_qcm[qubit].name, 'settings')

            #run calibration_qubit_states
            all_gnd_states, mean_gnd_states, all_exc_states, mean_exc_states = self.calibrate_qubit_states(qubit)
            # print(mean_gnd_states)
            # print(mean_exc_states)
            #TODO: Remove plot qubit states results
            # DEBUG: auto_calibrate_platform - Plot qubit states
            utils.plot_qubit_states(all_gnd_states, all_exc_states)
            self.save_config_parameter("mean_gnd_states", mean_gnd_states, 'characterization', 'single_qubit', qubit)
            self.save_config_parameter("mean_exc_states", mean_exc_states, 'characterization', 'single_qubit', qubit)

    def backup_config_file(self):
        import shutil
        from datetime import datetime

        settings_backups_folder = qibolab_folder / "calibration" / "data" / "settings_backups"
        settings_backups_folder.mkdir(parents=True, exist_ok=True)

        original = str(self.platform.runcard)
        original_file_name = pathlib.Path(original).name
        timestamp = datetime.now()
        timestamp = timestamp.strftime("%Y%m%d%H%M%S")
        destination_file_name = timestamp + '_' + original_file_name
        target = str(settings_backups_folder / destination_file_name)

        shutil.copyfile(original, target)

    def get_config_parameter(self, parameter, *keys):
        import os
        calibration_path = self.platform.runcard
        with open(calibration_path) as file:
            settings = yaml.safe_load(file)
        file.close()

        node = settings
        for key in keys:
            node = node.get(key)
        return node[parameter]

    def save_config_parameter(self, parameter, value, *keys):
        calibration_path = self.platform.runcard
        with open(calibration_path, "r") as file:
            settings = yaml.safe_load(file)
        file.close()

        node = settings
        for key in keys:
            node = node.get(key)
        node[parameter] = value

        # store latest timestamp
        import datetime
        settings['timestamp'] = datetime.datetime.utcnow()

        with open(calibration_path, "w") as file:
            settings = yaml.dump(settings, file, sort_keys=False, indent=4)
        file.close()

# help classes
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
        self.base_duration = qd_pulse.duration

    def set(self, value):
        # TODO: implement following condition
        #must be >= 4ns <= 65535
        #platform.delay_before_readout = value
        self.ro_pulse.start = self.base_duration + value


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

    def __init__(self, platform, sequence):
        self.platform = platform
        self.sequence = sequence

    def get(self):
        results = self.platform.execute_pulse_sequence(self.sequence)
        return list(list(results.values())[0].values())[0] #TODO: Replace with the particular acquisition
