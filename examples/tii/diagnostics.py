import pathlib
from qibolab.paths import qibolab_folder
import numpy as np
import matplotlib.pyplot as plt
import yaml
from qibolab import Platform
from qibolab.calibration import utils
from qibolab.calibration import fitting

# TODO: Have a look in the documentation of ``MeasurementControl``
from quantify_core.measurement import MeasurementControl
from quantify_core.measurement.control import Gettable, Settable
from quantify_core.data.handling import set_datadir
from scipy.signal import savgol_filter

from qibolab.pulses import Pulse, ReadoutPulse
from qibolab.circuit import PulseSequence
from qibolab.pulse_shapes import Rectangular, Gaussian


script_folder = pathlib.Path(__file__).parent

quantify_folder = qibolab_folder / "calibration" / "data" / "quantify"
quantify_folder.mkdir(parents=True, exist_ok=True)
set_datadir(quantify_folder)


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


def create_measurement_control(name):
    import os
    if os.environ.get("ENABLE_PLOTMON", True):
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
        return mc, plotmon, insmon
    # TODO: be able to choose which windows are opened and remember their sizes and dimensions 
   

class Diagnostics():

    def __init__(self, platform: Platform):
        self.platform = platform
        self.mc, self.pl, self.ins = create_measurement_control('Diagnostics')

    def load_settings(self):
        # Load diagnostics settings
        with open(script_folder / "diagnostics.yml", "r") as file:
            self.settings = yaml.safe_load(file)
        self.__dict__.update(self.settings)

    def reload_settings(self):
        self.load_settings()

    def run_resonator_spectroscopy(self, qubit=1):
        # TODO: replace lo_qrm with generic access to the relevant instrument after searching for it based on the qubit/channel
        platform = self.platform
        platform.reload_settings()
        mc = self.mc
        ro_channel = platform.settings['qubit_channel_map'][qubit][0]

        ps = platform.settings['shared_settings']
        ro_pulse_settings = ps['readout_pulse']
        ro_pulse = ReadoutPulse(**ro_pulse_settings, channel = ro_channel)
        sequence = PulseSequence()
        sequence.add(ro_pulse)

        self.reload_settings()
        self.__dict__.update(self.settings['resonator_spectroscopy'])
        self.pl.tuids_max_num(self.max_num_plots)

        #Fast Sweep
        if (self.software_averages !=0):
            scanrange = variable_resolution_scanrange(self.lowres_width, self.lowres_step, self.highres_width, self.highres_step)
            mc.settables(platform.lo_qrm.device.frequency)
            mc.setpoints(scanrange + platform.lo_qrm.frequency)
            mc.gettables(Gettable(ROController(platform, sequence)))
            platform.start() 
            platform.lo_qcm.off()
            dataset = mc.run("Resonator Spectroscopy Fast", soft_avg=self.software_averages)
            platform.stop()
            platform.lo_qrm.frequency = (dataset['x0'].values[dataset['y0'].argmax().values])
            avg_min_voltage = np.mean(dataset['y0'].values[:(self.lowres_width//self.lowres_step)]) * 1e6

        # Precision Sweep
        if (self.software_averages_precision !=0):
            scanrange = np.arange(-self.precision_width, self.precision_width, self.precision_step)
            mc.settables(platform.lo_qrm.device.frequency)
            mc.setpoints(scanrange + platform.lo_qrm.frequency)
            mc.gettables(Gettable(ROController(platform, sequence)))
            platform.start() 
            platform.lo_qcm.off()
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
        ro_channel = platform.settings['qubit_channel_map'][qubit][0]
        qc_channel = platform.settings['qubit_channel_map'][qubit][1]

        ps = platform.settings['shared_settings']
        qc_pulse_settings = ps['qc_spectroscopy_pulse']
        qc_pulse = Pulse(**qc_pulse_settings, channel = qc_channel)
        ro_pulse_settings = ps['readout_pulse']
        ro_pulse = ReadoutPulse(**ro_pulse_settings, channel = ro_channel)
        sequence = PulseSequence()
        sequence.add(qc_pulse)
        sequence.add(ro_pulse)

        self.reload_settings()
        self.__dict__.update(self.settings['qubit_spectroscopy'])
        self.pl.tuids_max_num(self.max_num_plots)
        
        
        # Fast Sweep
        if (self.software_averages !=0):
            lo_qcm_frequency = platform.lo_qcm.frequency
            fast_sweep_scan_range = np.arange(self.fast_start, self.fast_end, self.fast_step)
            mc.settables(platform.lo_qcm.device.frequency)
            mc.setpoints(fast_sweep_scan_range + platform.lo_qcm.frequency)
            mc.gettables(Gettable(ROController(platform, sequence)))
            platform.start() 
            dataset = mc.run("Qubit Spectroscopy Fast", soft_avg=self.software_averages)
            platform.stop()


        # Precision Sweep
        if (self.software_averages_precision !=0):
            platform.lo_qcm.frequency = lo_qcm_frequency
            precision_sweep_scan_range = np.arange(self.precision_start, self.precision_end, self.precision_step)
            mc.settables(platform.lo_qcm.device.frequency)
            mc.setpoints(precision_sweep_scan_range + platform.lo_qcm.frequency)
            mc.gettables(Gettable(ROController(platform, sequence)))
            platform.start() 
            dataset = mc.run("Qubit Spectroscopy Precision", soft_avg=self.software_averages_precision)
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

    def run_rabi_pulse_length(self, qubit=1):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc
        ro_channel = platform.settings['qubit_channel_map'][qubit][0]
        qc_channel = platform.settings['qubit_channel_map'][qubit][1]

        ps = platform.settings['shared_settings']
        qc_pulse_settings = ps['qc_spectroscopy_pulse']
        qc_pulse = Pulse(**qc_pulse_settings, channel = qc_channel)
        ro_pulse_settings = ps['readout_pulse']
        ro_pulse = ReadoutPulse(**ro_pulse_settings, channel = ro_channel)
        sequence = PulseSequence()
        sequence.add(qc_pulse)
        sequence.add(ro_pulse)

        self.reload_settings()
        self.__dict__.update(self.settings['rabi_pulse_length'])
        self.pl.tuids_max_num(self.max_num_plots)


        mc.settables(Settable(QCPulseLengthParameter(ro_pulse, qc_pulse)))
        mc.setpoints(np.arange(self.pulse_duration_start, self.pulse_duration_end, self.pulse_duration_step))
        mc.gettables(Gettable(ROController(platform, sequence)))
        platform.start()
        dataset = mc.run('Rabi Pulse Length', soft_avg = self.software_averages)
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

    def run_rabi_pulse_gain(self, platform, mc, resonator_freq, qubit_freq, sequence, qc_pulse, ro_pulse):

        """#qubit pulse duration=200
        platform.lo_qrm.frequency = (resonator_freq - ro_pulse.frequency)
        platform.lo_qcm.frequency = (qubit_freq + qc_pulse.frequency)
        platform.software_averages = 3
        mc.settables(Settable(QCPulseGainParameter(platform.qcm)))
        mc.setpoints(np.arange(0, 100, 10))
        mc.gettables(Gettable(ROController(platform, sequence)))
        platform.start()
        dataset = mc.run('Rabi Pulse Gain', soft_avg = platform.software_averages)
        platform.stop()
        
        return dataset
        """
        raise NotImplementedError

    def run_rabi_pulse_length_and_gain(self, platform, mc, resonator_freq, qubit_freq, sequence, qc_pulse, ro_pulse):
        """
        platform.lo_qrm.frequency = (resonator_freq - ro_pulse.frequency)
        platform.lo_qcm.frequency = (qubit_freq + qc_pulse.frequency)
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
        """
        raise NotImplementedError

    def run_rabi_pulse_length_and_amplitude(self, platform, mc, resonator_freq, qubit_freq, sequence, qc_pulse, ro_pulse):
        """
        platform.lo_qrm.frequency = (resonator_freq - ro_pulse.frequency)
        platform.lo_qcm.frequency = (qubit_freq + qc_pulse.frequency)
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
        """
        raise NotImplementedError

    # T1: RX(pi) - wait t(rotates z) - readout
    def run_t1(self, qubit=1):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc
        qc_channel = platform.settings['qubit_channel_map'][1][1]
        ro_channel = platform.settings['qubit_channel_map'][1][0]
        
        RX_pulse = platform.settings['native_gates']['single_qubit'][qubit]['RX']['pulse_sequence'][0]
        qc_pi_pulse = Pulse(**RX_pulse, channel = qc_channel)
        ps = platform.settings['shared_settings']
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
        qc_channel = platform.settings['qubit_channel_map'][1][1]
        ro_channel = platform.settings['qubit_channel_map'][1][0]
        
        RX_pulse = platform.settings['native_gates']['single_qubit'][qubit]['RX']['pulse_sequence'][0]
        RX90_pulse = RX_pulse.copy()
        RX90_pulse.update({'amplitude': RX_pulse['amplitude']/2})
        
        qc_pi_half_pulse_1 = Pulse(**RX90_pulse, channel = qc_channel)
        qc_pi_half_pulse_2 = Pulse(**RX90_pulse, channel = qc_channel)
        qc_pi_half_pulse_2.start = qc_pi_half_pulse_1.start + qc_pi_half_pulse_1.duration
        
        ps = platform.settings['shared_settings']
        ro_pulse_settings = ps['readout_pulse']
        ro_pulse = ReadoutPulse(**ro_pulse_settings, channel = ro_channel)
        sequence = PulseSequence()
        sequence.add(qc_pi_half_pulse_1)
        sequence.add(qc_pi_half_pulse_2)
        sequence.add(ro_pulse)
        
        self.reload_settings()
        self.__dict__.update(self.settings['ramsey'])
        self.pl.tuids_max_num(self.max_num_plots)

        mc.settables(Settable(RamseyWaitParameter(ro_pulse, qc_pi_half_pulse_2, RX_pulse['duration'])))
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

        return t2, smooth_dataset, dataset

    # Spin Echo: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - readout
    def run_spin_echo(self):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc
        qc_channel = platform.settings['qubit_channel_map'][1][1]
        ro_channel = platform.settings['qubit_channel_map'][1][0]
        
        ps = platform.settings['shared_settings']
        start = 0
        frequency = ps['pi_pulse_frequency']
        amplitude = ps['pi_pulse_amplitude']
        duration = ps['pi_pulse_duration']
        phase = 0
        shape =ps['pi_pulse_shape']
        qc_pi_half_pulse = Pulse(start, duration, amplitude/2, frequency, phase, shape)
        qc_pi_pulse = Pulse(qc_pi_half_pulse.start + qc_pi_half_pulse.duration, duration, amplitude, frequency, phase, shape)
        
        ro_pulse_shape =ps['readout_pulse']['shape']
        ro_pulse_settings = ps['readout_pulse']
        ro_pulse = ReadoutPulse(**ro_pulse_settings, channel = ro_channel, shape = ro_pulse_shape)
        sequence = PulseSequence()
        sequence.add(qc_pi_half_pulse)
        sequence.add(qc_pi_pulse)
        sequence.add(ro_pulse)
        
        ds = self.load_settings()
        self.pl.tuids_max_num(ds['max_num_plots'])
        software_averages = ds['software_averages']
        ds = ds['spin_echo']
        delay_between_pulses_start = ds['delay_between_pulses_start']
        delay_between_pulses_end = ds['delay_between_pulses_end']
        delay_between_pulses_step = ds['delay_between_pulses_step']

        mc.settables(Settable(SpinEchoWaitParameter(ro_pulse, qc_pi_pulse, platform.settings['shared_settings']['pi_pulse_duration'])))
        mc.setpoints(np.arange(delay_between_pulses_start, delay_between_pulses_end, delay_between_pulses_step))
        mc.gettables(Gettable(ROController(platform, sequence)))
        platform.start()
        dataset = mc.run('Spin Echo', soft_avg = software_averages)
        platform.stop()
        
        # Fitting

        return dataset

    # Spin Echo 3 Pulses: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - RX(pi/2) - readout
    def run_spin_echo_3pulses(self):
        
        platform = self.platform
        platform.reload_settings()
        mc = self.mc
        qc_channel = platform.settings['qubit_channel_map'][1][1]
        ro_channel = platform.settings['qubit_channel_map'][1][0]
        
        ps = platform.settings['shared_settings']
        start = 0
        frequency = ps['pi_pulse_frequency']
        amplitude = ps['pi_pulse_amplitude']
        duration = ps['pi_pulse_duration']
        phase = 0
        shape =ps['pi_pulse_shape']
        qc_pi_half_pulse_1 = Pulse(start, duration, amplitude/2, frequency, phase, shape)
        qc_pi_pulse = Pulse(qc_pi_half_pulse_1.start + qc_pi_half_pulse_1.duration, duration, amplitude, frequency, phase, shape)
        qc_pi_half_pulse_2 = Pulse(qc_pi_pulse.start + qc_pi_pulse.duration, duration, amplitude/2, frequency, phase, shape)
        
        ro_pulse_shape =ps['readout_pulse']['shape']
        ro_pulse_settings = ps['readout_pulse']
        ro_pulse = ReadoutPulse(**ro_pulse_settings, channel = ro_channel, shape = ro_pulse_shape)
        sequence = PulseSequence()
        sequence.add(qc_pi_half_pulse_1)
        sequence.add(qc_pi_pulse)
        sequence.add(qc_pi_half_pulse_2)
        sequence.add(ro_pulse)
        
        ds = self.load_settings()
        self.pl.tuids_max_num(ds['max_num_plots'])
        software_averages = ds['software_averages']
        ds = ds['spin_echo_3pulses']        
        delay_between_pulses_start = ds['delay_between_pulses_start']
        delay_between_pulses_end = ds['delay_between_pulses_end']
        delay_between_pulses_step = ds['delay_between_pulses_step']


        mc.settables(SpinEcho3PWaitParameter(ro_pulse, qc_pi_pulse, qc_pi_half_pulse_2, platform.settings['shared_settings']['pi_pulse_duration']))
        mc.setpoints(np.arange(delay_between_pulses_start, delay_between_pulses_end, delay_between_pulses_step))
        mc.gettables(Gettable(ROController(platform, sequence)))
        platform.start()
        dataset = mc.run('Spin Echo 3 Pulses', soft_avg = software_averages)
        platform.stop()

        return dataset

    def run_shifted_resonator_spectroscopy(self, platform, mc, resonator_freq, qubit_freq, sequence, qc_pulse, ro_pulse,
                                        lowres_width, lowres_step, highres_width, highres_step,
                                        precision_width, precision_step):

        platform.lo_qrm.frequency = (resonator_freq - ro_pulse.frequency)
        platform.lo_qcm.frequency = (qubit_freq + qc_pulse.frequency)

        # Fast Sweep
        platform.software_averages = 1
        scanrange = variable_resolution_scanrange(lowres_width, lowres_step, highres_width, highres_step)
        mc.settables(platform.lo_qrm.device.frequency)
        mc.setpoints(scanrange + platform.lo_qrm.frequency)
        mc.gettables(Gettable(ROController(platform, sequence)))
        platform.start()
        dataset = mc.run("Resonator Spectroscopy Shifted Fast", soft_avg=platform.software_averages)
        platform.stop()

        shifted_LO_frequency = dataset['x0'].values[dataset['y0'].argmax().values]

        # Precision Sweep
        platform.software_averages = 1
        scanrange = np.arange(-precision_width, precision_width, precision_step)
        mc.settables(platform.lo_qrm.device.frequency)
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


    def run_punchout(self):     
        platform = self.platform
        platform.reload_settings()
        mc = self.mc
        qc_channel = platform.settings['qubit_channel_map'][1][1]
        ro_channel = platform.settings['qubit_channel_map'][1][0]
        
        ps = platform.settings['shared_settings']
        qc_pulse_shape =ps['qc_spectroscopy_pulse']['shape']
        qc_pulse_settings = ps['qc_spectroscopy_pulse']
        qc_pulse = Pulse(**qc_pulse_settings, channel = qc_channel, shape = qc_pulse_shape)
        ro_pulse_shape =ps['readout_pulse']['shape']
        ro_pulse_settings = ps['readout_pulse']
        ro_pulse = ReadoutPulse(**ro_pulse_settings, channel = ro_channel, shape = ro_pulse_shape)
        sequence = PulseSequence()
        sequence.add(qc_pulse)
        sequence.add(ro_pulse)

        ds = self.load_settings()['punchout']
        self.pl.tuids_max_num(ds['max_num_plots'])
        software_averages = ds['software_averages']
        ds = ds['punchout']
        precision_width = ds['precision_width']
        precision_step = ds['precision_step']


        scanrange = np.arange(-precision_width, precision_width, precision_step)
        scanrange = scanrange + platform.lo_qrm.frequency

        mc.settables([Settable(platform.lo_qrm.device.frequency), Settable(QRPulseGainParameter(platform.qrm))])
        setpoints_gain = np.arange(10, 100, 10)
        mc.setpoints_grid([scanrange, setpoints_gain])
        mc.gettables(Gettable(ROController(platform, sequence)))
        platform.start() 
        platform.lo_qcm.off()
        dataset = mc.run("Punchout", soft_avg=software_averages)
        platform.stop()

        # Fitting
        smooth_dataset = savgol_filter(dataset['y0'].values, 25, 2)
        #FIXME: Code Lorentzian fitting for cavity spec and punchout
        resonator_freq = dataset['x0'].values[dataset['y0'].argmax().values]+ro_pulse.frequency 
        print(f"\nResonator Frequency = {resonator_freq}")
        print(f"\nResonator LO Frequency  = {resonator_freq - ro_pulse.frequency}")

        return resonator_freq, smooth_dataset, dataset










    def callibrate_qubit_states(self):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc
        qc_channel = platform.settings['qubit_channel_map'][1][1]
        ro_channel = platform.settings['qubit_channel_map'][1][0]

        ps = platform.settings['shared_settings']
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
            qubit_state = platform.execute_pulse_sequence(exc_sequence, nshots)
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
            qubit_state = platform.execute_pulse_sequence(gnd_sequence, nshots)
            qubit_state = list(list(qubit_state.values())[0].values())[0]
            print(f"Finished gnd single shot execution  {i}")
            #Compose complex point from i, q obtained from execution
            point = complex(qubit_state[2], qubit_state[3])
            all_gnd_states.append(point)
        platform.stop()

        return all_gnd_states, np.mean(all_gnd_states), all_exc_states, np.mean(all_exc_states)


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
        self.qc2_pulse.start = self.pi_pulse_length  + value
        self.ro_pulse.start = self.pi_pulse_length * 2 + value + 4

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
        self.qc2_pulse.start = self.pi_pulse_length + value
        self.ro_pulse.start = 2 * self.pi_pulse_length + 2 * value + 4

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
        self.qc2_pulse.start = self.pi_pulse_length + value
        self.qc3_pulse.start = 2 * self.pi_pulse_length + 2 * value
        self.ro_pulse.start = 3 * self.pi_pulse_length + 2 * value + 4

class QRPulseGainParameter():

    label = 'Qubit Readout Gain'
    unit = '%'
    name = 'ro_pulse_gain'

    def __init__(self, qrm):
        self.qrm = qrm

    def set(self,value):
        self.qrm.gain = value / 100

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
        return list(list(results.values())[0].values())[0]