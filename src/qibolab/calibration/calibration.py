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

        lo_qrm = platform.lo_qrm[qubit]
        lo_qcm = platform.lo_qcm[qubit]

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

        #Fast Sweep
        if (self.software_averages !=0):
            scanrange = utils.variable_resolution_scanrange(self.lowres_width, self.lowres_step, self.highres_width, self.highres_step)
            mc.settables(lo_qrm.settable_frequency)
            mc.setpoints(scanrange + lo_qrm.frequency)
            mc.gettables(Gettable(ROController(platform, sequence, qubit)))
            platform.start() 
            lo_qcm.off()
            dataset = mc.run("Resonator Spectroscopy Fast", soft_avg=self.software_averages)
            platform.stop()
            lo_qrm.frequency = dataset['x0'].values[dataset['y0'].argmax().values]
            avg_min_voltage = np.mean(dataset['y0'].values[:(self.lowres_width//self.lowres_step)]) * 1e6

        # Precision Sweep
        if (self.software_averages_precision !=0):
            scanrange = np.arange(-self.precision_width, self.precision_width, self.precision_step)
            mc.settables(lo_qrm.settable_frequency)
            mc.setpoints(scanrange + lo_qrm.frequency)
            mc.gettables(Gettable(ROController(platform, sequence, qubit)))
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

    def run_qubit_spectroscopy(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        lo_qcm = platform.lo_qcm[qubit]

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
        
        # Fast Sweep
        if (self.software_averages !=0):
            lo_qcm_frequency = lo_qcm.frequency
            fast_sweep_scan_range = np.arange(self.fast_start, self.fast_end, self.fast_step)
            mc.settables(lo_qcm.settable_frequency)
            mc.setpoints(fast_sweep_scan_range + lo_qcm.frequency)
            mc.gettables(Gettable(ROController(platform, sequence, qubit)))
            platform.start() 
            dataset = mc.run("Qubit Spectroscopy Fast", soft_avg=self.software_averages)
            platform.stop()

        # Precision Sweep
        if (self.software_averages_precision !=0):
            lo_qcm.frequency = lo_qcm_frequency
            precision_sweep_scan_range = np.arange(self.precision_start, self.precision_end, self.precision_step)
            mc.settables(lo_qcm.settable_frequency)
            mc.setpoints(precision_sweep_scan_range + lo_qcm.frequency)
            mc.gettables(Gettable(ROController(platform, sequence, qubit)))
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

        mc.settables(Settable(QCPulseLengthParameter(ro_pulse, qd_pulse)))
        mc.setpoints(np.arange(self.pulse_duration_start, self.pulse_duration_end, self.pulse_duration_step))
        mc.gettables(Gettable(ROController(platform, sequence, qubit)))
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

        mc.settables(Settable(T1WaitParameter(ro_pulse, RX_pulse)))
        mc.setpoints(np.arange( self.delay_before_readout_start,
                                self.delay_before_readout_end,
                                self.delay_before_readout_step))
        mc.gettables(Gettable(ROController(platform, sequence, qubit)))
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

        mc.settables(Settable(RamseyWaitParameter(ro_pulse, RX90_pulse2)))
        mc.setpoints(np.arange(self.delay_between_pulses_start, self.delay_between_pulses_end, self.delay_between_pulses_step))
        mc.gettables(Gettable(ROController(platform, sequence, qubit)))
        platform.start()
        dataset = mc.run('Ramsey', soft_avg = self.software_averages)
        platform.stop()

        # Fitting
        smooth_dataset, delta_frequency, t2 = fitting.ramsey_fit(dataset)
        utils.plot(smooth_dataset, dataset, "Ramsey", 1)
        print(f"\nDelta Frequency = {delta_frequency}")
        print(f"\nT2 = {t2} ns")

        return delta_frequency, t2, smooth_dataset, dataset

    def calibrate_qubit_states(self, qubit=0):
        platform = self.platform
        platform.reload_settings()

        lo_qcm = platform.lo_qcm[qubit]
        
        self.reload_settings()
        self.niter = self.settings['calibrate_qubit_states']['niter']
        
        #create exc and gnd pulses 
        exc_sequence = PulseSequence()
        RX_pulse = platform.RX_pulse(qubit, start = 0)
        ro_pulse = platform.qubit_readout_pulse(qubit, start = RX_pulse.duration)
        exc_sequence.add(RX_pulse)
        exc_sequence.add(ro_pulse)

        platform.start()
        #Exectue niter single exc shots
        all_exc_states = []
        for i in range(self.niter):
            print(f"Starting exc state calibration {i}")
            qubit_state = platform.execute_pulse_sequence(exc_sequence, nshots = 1) # TODO: Improve the speed of this with binning
            qubit_state = list(list(qubit_state.values())[0].values())[0]
            print(f"Finished exc single shot execution  {i}")
            #Compose complex point from i, q obtained from execution
            point = complex(qubit_state[2], qubit_state[3])
            all_exc_states.append(point)
        platform.stop()

        gnd_sequence = PulseSequence()
        ro_pulse = platform.qubit_readout_pulse(qubit, start = 0)
        gnd_sequence.add(ro_pulse)

        #Exectue niter single gnd shots
        platform.start()
        lo_qcm.off()
        all_gnd_states = []
        for i in range(self.niter):
            print(f"Starting gnd state calibration  {i}")
            qubit_state = platform.execute_pulse_sequence(gnd_sequence, 1) # TODO: Improve the speed of this with binning
            qubit_state = list(list(qubit_state.values())[0].values())[0]
            print(f"Finished gnd single shot execution  {i}")
            #Compose complex point from i, q obtained from execution
            point = complex(qubit_state[2], qubit_state[3])
            all_gnd_states.append(point)
        platform.stop()

        return all_gnd_states, np.mean(all_gnd_states), all_exc_states, np.mean(all_exc_states)

    def fromReadout(self, readout, min_voltage, max_voltage):
        norm = max_voltage - min_voltage
        normalized_voltage =  (readout[0] * 1e6 - min_voltage) / norm
        return normalized_voltage

    def toSequence(self, gates, qubit):   
        #read settings
        platform = self.platform
        platform.reload_settings()

        ps = platform.settings['settings']
                
        sequenceDuration = 0
        sequence = PulseSequence()
        
        start_pulse = 0
        for gate in gates:    
            if (gate == "I"):
                print("Transforming to sequence I gate")
                duration = 0
            
            if (gate == "RX(pi)"):
                print("Transforming to sequence RX(pi) gate")
                RX_pulse = platform.RX_pulse(qubit, start = start_pulse)
                duration = RX_pulse.duration
                sequence.add(RX_pulse)

            if (gate == "RX(pi/2)"):
                print("Transforming to sequence RX(pi/2) gate")
                RX90_pulse = platform.RX90_pulse(qubit, start = start_pulse)
                duration = RX90_pulse.duration
                sequence.add(RX90_pulse)

            if (gate == "RY(pi)"):
                print("Transforming to sequence RY(pi) gate")
                RY_pulse = platform.RX_pulse(qubit, start = start_pulse, phase = np.pi)
                duration = RY_pulse.duration
                sequence.add(RY_pulse)

            if (gate == "RY(pi/2)"):
                print("Transforming to sequence RY(pi/2) gate")
                RY90_pulse = platform.RX90_pulse(qubit, start = start_pulse, phase = np.pi)
                duration = RY90_pulse.duration
                sequence.add(RY90_pulse)
            
            sequenceDuration = sequenceDuration + duration
            start_pulse = duration

        #RO pulse starting just after pair of gates
        ro_pulse = platform.qubit_readout_pulse(qubit, start = sequenceDuration + 4)
        sequence.add(ro_pulse)
        
        return sequence

    def allXY(self, qubit):
        platform = self.platform
        platform.reload_settings()

        #allXY rotations
        gatelist = [
            ["I","I"], 
            ["RX(pi)","RX(pi)"],
            ["RY(pi)","RY(pi)"],    
            ["RX(pi)","RY(pi)"],        
            ["RY(pi)","RX(pi)"],
            ["RX(pi/2)","I"],        
            ["RY(pi/2)","I"],            
            ["RX(pi/2)","RY(pi/2)"],            
            ["RX(pi/2)","RY(pi/2)"],                
            ["RX(pi/2)","RY(pi)"],                
            ["RY(pi/2)","RX(pi)"],                
            ["RX(pi)","RY(pi/2)"],                
            ["RX(pi)","RX(pi/2)"],                
            ["RX(pi/2)","RX(pi)"],                            
            ["RX(pi)","RX(pi/2)"],                
            ["RY(pi/2)","RY(pi)"],                
            ["RY(pi)","RY(pi/2)"],                
            ["RX(pi)","I"],  
            ["RY(pi)","I"],                
            ["RX(pi/2)","RX(pi/2)"],                
            ["RY(pi/2)","RY(pi/2)"]                
           ]

        results = []
        gateNumber = []
        min_voltage = platform.settings['characterization']['single_qubit'][qubit]['rabi_oscillations_pi_pulse_min_voltage']
        max_voltage = platform.settings['characterization']['single_qubit'][qubit]['resonator_spectroscopy_max_ro_voltage']
        n = 0 
        for gates in gatelist:
            #transform gate string to pulseSequence
            seq = self.toSequence(gates, qubit)      
            #Execute PulseSequence defined by gates
            platform.start()
            state = platform.execute_pulse_sequence(seq, nshots=1024)
            state = list(list(state.values())[0].values())[0]
            platform.stop()
            #transform readout I and Q into probabilities
            res = self.fromReadout(state, min_voltage, max_voltage)
            res = (2 * res) - 1
            results.append(res)
            gateNumber.append(n)
            n=n+1

        return results, gateNumber

    # Ramsey: RX(pi/2) - wait t(rotates z) - RX(pi/2) - readout
    def run_ramsey_freq(self, qubit):
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
        self.t_start = self.settings['ramsey_freq']['t_start']
        self.t_end = self.settings['ramsey_freq']['t_end']
        self.t_step = self.settings['ramsey_freq']['t_step']
        self.N_osc = self.settings['ramsey_freq']['N_osc']

        stop = False        
        self.pl.tuids_max_num(self.max_num_plots)
        

        for t_max in self.t_end:
            if (stop == False):
                offset_freq = (self.N_osc / t_max * 1e9) #Hz
                t_range = np.arange(self.t_start, t_max, self.t_step)
                mc.settables(Settable(RamseyFreqWaitParameter(ro_pulse, RX90_pulse2, offset_freq)))
                mc.setpoints(t_range)
                mc.gettables(Gettable(ROController(platform, sequence)))
                platform.start()
                dataset = mc.run('Ramsey_freq', soft_avg = self.software_averages)
                platform.stop()

                # Fitting
                smooth_dataset, delta_fitting, new_t2 = fitting.ramsey_freq_fit(dataset)
 
                delta_phys = (delta_fitting * 1e9) - offset_freq
                
                actual_qubit_freq = platform.settings['characterization']['single_qubit'][qubit]['qubit_freq']
                T2 = platform.settings['characterization']['single_qubit'][qubit]['T2']

                #if ((new_t2 * 3.5) > t_max):
                if (new_t2 > T2):
                    qubit_freq = actual_qubit_freq + delta_phys 
                    # self.save_config_parameter("settings", "", "qubit_freq", float(qubit_freq))
                    # self.save_config_parameter("LO_QCM_settings", "", "frequency", float(qubit_freq + 200_000_000))
                    # self.save_config_parameter("settings", "", "T2", float(new_t2))
                else:
                    stop = True

                platform.reload_settings()
                # FIXME: The way this routine is coded the new_T2 and delta_phys returned are not the optimal.
        return new_t2, delta_phys, smooth_dataset, dataset
   
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
            lo_qrm_frequency = int(resonator_freq - platform.settings['native_gates']['single_qubit'][qubit]['MZ']['frequency'])
            self.save_config_parameter("frequency", lo_qrm_frequency, 'instruments', platform.lo_qrm[qubit].name, 'settings') # TODO: cambiar IF hardcoded

            # run and save qubit spectroscopy calibration
            qubit_freq, min_ro_voltage, smooth_dataset, dataset = self.run_qubit_spectroscopy(qubit)
            self.save_config_parameter("qubit_freq", int(qubit_freq), 'characterization', 'single_qubit', qubit)
            RX_pulse_sequence = platform.settings['native_gates']['single_qubit'][qubit]['RX']['pulse_sequence']
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
            RX_pulse_sequence = platform.settings['native_gates']['single_qubit'][qubit]['RX']['pulse_sequence']
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

class RamseyFreqWaitParameter():
    label = 'Time'
    unit = 'ns'
    name = 'ramsey_freq'
    initial_value = 0

    def __init__(self, ro_pulse,  qc2_pulse, offset_freq):
        self.ro_pulse = ro_pulse
        self.qc2_pulse = qc2_pulse
        self.pi_pulse_length = qc2_pulse.duration
        self.offset_freq = offset_freq

    def set(self, value):
        self.ro_pulse.start = self.pi_pulse_length * 2 + value + 4
        self.qc2_pulse.start = self.pi_pulse_length + value
        value_phase = (value * 1e-9) * 2 * np.pi * self.offset_freq
        self.qc2_pulse.phase = value_phase

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
