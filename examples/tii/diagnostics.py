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
<<<<<<< HEAD
from qibolab.pulse import Pulse, ReadoutPulse
from qibolab.pulse.pulse_shape import Rectangular, Gaussian, Drag
from qibolab.circuit import PulseSequence
=======
from qibolab.pulses import PulseSequence, Pulse, ReadoutPulse, Rectangular, Gaussian, Drag
>>>>>>> alvaro/qblox_upgrade_0.6.1


class Diagnostics():

    def __init__(self, platform: Platform):
        self.platform = platform
        self.mc, self.pl, self.ins = utils.create_measurement_control('Diagnostics')

    def load_settings(self):
        # Load diagnostics settings
        script_folder = pathlib.Path(__file__).parent
        with open(script_folder / "diagnostics.yml", "r") as file:
            self.settings = yaml.safe_load(file)
            self.software_averages = self.settings['software_averages']
            self.software_averages_precision = self.settings['software_averages_precision']
            self.max_num_plots = self.settings['max_num_plots']

    def reload_settings(self):
        self.load_settings()

    def run_rabi_pulse_gain(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        qcm = platform.qcm[qubit]

        sequence = PulseSequence()
        qd_pulse = platform.qubit_readout_pulse(qubit, start = 0, duration = 60) # TODO: To diagnostics.yml?
        ro_pulse = platform.qubit_readout_pulse(qubit, start = qd_pulse.duration)
        sequence.add(qd_pulse)
        sequence.add(ro_pulse)

        self.reload_settings()
        self.pulse_gain_start = self.settings['rabi_pulse_gain']['pulse_gain_start']
        self.pulse_gain_end = self.settings['rabi_pulse_gain']['pulse_gain_end']
        self.pulse_gain_step = self.settings['rabi_pulse_gain']['pulse_gain_step']

        self.pl.tuids_max_num(self.max_num_plots)

        mc.settables(Settable(QCPulseGainParameter(qcm)))
        mc.setpoints(np.arange(self.pulse_gain_start, self.pulse_gain_end, self.pulse_gain_step))
        mc.gettables(Gettable(ROController(platform, sequence, qubit)))
        platform.start()
        dataset = mc.run('Rabi Pulse Gain', soft_avg = self.software_averages)
        platform.stop()

        return dataset

    def run_rabi_pulse_length_and_gain(self, qubit=0):
        """
        platform.lo_qrm.frequency = (resonator_freq - ro_pulse.frequency)
        platform.lo_qcm.frequency = (qubit_freq + qd_pulse.frequency)
        platform.software_averages = 1
        mc.settables([Settable(QCPulseLengthParameter(ro_pulse, qd_pulse)),
                    Settable(QCPulseGainParameter(platform.qcm))])
        setpoints_length = np.arange(1, 400, 10)
        setpoints_gain = np.arange(0, 20, 1)
        mc.setpoints_grid([setpoints_length, setpoints_gain])
        mc.gettables(Gettable(ROController(platform, sequence, qubit)))
        platform.start()
        dataset = mc.run('Rabi Pulse Length and Gain', soft_avg = platform.software_averages)
        # Analyse data to look for the smallest qd_pulse length that renders off-resonance amplitude, determine corresponding pi_pulse gain
        # platform.pi_pulse_length =
        # platform.pi_pulse_gain =
        platform.stop()
        
        return dataset
        """
        raise NotImplementedError

    def run_rabi_pulse_length_and_amplitude(self, qubit=0):
        """
        platform.lo_qrm.frequency = (resonator_freq - ro_pulse.frequency)
        platform.lo_qcm.frequency = (qubit_freq + qd_pulse.frequency)
        platform.software_averages = 1
        mc.settables([Settable(QCPulseLengthParameter(ro_pulse, qd_pulse)),
                    Settable(QCPulseAmplitudeParameter(qd_pulse))])
        setpoints_length = np.arange(1, 1000, 2)
        setpoints_amplitude = np.arange(0, 100, 2)
        mc.setpoints_grid([setpoints_length, setpoints_amplitude])
        mc.gettables(Gettable(ROController(platform, sequence, qubit)))
        platform.start()
        dataset = mc.run('Rabi Pulse Length and Gain', soft_avg = platform.software_averages)
        # Analyse data to look for the smallest qd_pulse length that renders off-resonance amplitude, determine corresponding pi_pulse gain
        # platform.pi_pulse_length =
        # platform.pi_pulse_gain =
        platform.stop()

        return dataset
        """
        raise NotImplementedError

    # Spin Echo: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - readout
    def run_spin_echo(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        sequence = PulseSequence()
        RX90_pulse = platform.RX90_pulse(qubit, start = 0)
        RX_pulse = platform.RX_pulse(qubit, start = RX90_pulse.duration)
        ro_pulse = platform.qubit_readout_pulse(qubit, start = RX_pulse.start + RX_pulse.duration)
        sequence.add(RX90_pulse)
        sequence.add(RX_pulse)
        sequence.add(ro_pulse)

        self.reload_settings()
        self.delay_between_pulses_start = self.settings['spin_echo']['delay_between_pulses_start']
        self.delay_between_pulses_end = self.settings['spin_echo']['delay_between_pulses_end']
        self.delay_between_pulses_step = self.settings['spin_echo']['delay_between_pulses_step']

        self.pl.tuids_max_num(self.max_num_plots)

        mc.settables(Settable(SpinEchoWaitParameter(ro_pulse, RX_pulse, platform.settings['settings']['pi_pulse_duration'])))
        mc.setpoints(np.arange(self.delay_between_pulses_start, self.delay_between_pulses_end, self.delay_between_pulses_step))
        mc.gettables(Gettable(ROController(platform, sequence, qubit)))
        platform.start()
        dataset = mc.run('Spin Echo', soft_avg = self.software_averages)
        platform.stop()
        
        # Fitting

        return dataset

    # Spin Echo 3 Pulses: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - RX(pi/2) - readout
    def run_spin_echo_3pulses(self, qubit=0):
        
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        sequence = PulseSequence()
        RX90_pulse1 = platform.RX90_pulse(qubit, start = 0)
        RX_pulse = platform.RX_pulse(qubit, start = RX90_pulse1.duration)
        RX90_pulse2 = platform.RX90_pulse(qubit, start = RX_pulse.start + RX_pulse.duration)
        ro_pulse = platform.qubit_readout_pulse(qubit, start = RX90_pulse2.start + RX90_pulse2.duration)
        sequence.add(RX90_pulse1)
        sequence.add(RX_pulse)
        sequence.add(RX90_pulse2)
        sequence.add(ro_pulse)
        
        self.reload_settings()
        self.delay_between_pulses_start = self.settings['spin_echo_3pulses']['delay_between_pulses_start']
        self.delay_between_pulses_end = self.settings['spin_echo_3pulses']['delay_between_pulses_end']
        self.delay_between_pulses_step = self.settings['spin_echo_3pulses']['delay_between_pulses_step']

        self.pl.tuids_max_num(self.max_num_plots)

        mc.settables(SpinEcho3PWaitParameter(ro_pulse, RX_pulse, RX90_pulse2, platform.settings['settings']['pi_pulse_duration']))
        mc.setpoints(np.arange(self.delay_between_pulses_start, self.delay_between_pulses_end, self.delay_between_pulses_step))
        mc.gettables(Gettable(ROController(platform, sequence, qubit)))
        platform.start()
        dataset = mc.run('Spin Echo 3 Pulses', soft_avg = self.software_averages)
        platform.stop()

        return dataset

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


class QCPulseGainParameter():

    label = 'Qubit Control Gain'
    unit = '%'
    name = 'qd_pulse_gain'

    def __init__(self, qcm):
        self.qcm = qcm

    def set(self,value):
        self.qcm.gain = value / 100


class QCPulseAmplitudeParameter():

    label = 'Qubit Control Pulse Amplitude'
    unit = '%'
    name = 'qd_pulse_amplitude'

    def __init__(self, qd_pulse):
        self.qd_pulse = qd_pulse

    def set(self, value):
        self.qd_pulse.amplitude = value / 100


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
        self.ro_pulse.start = 2 * self.pi_pulse_length + 2 * value


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
        self.ro_pulse.start = 3 * self.pi_pulse_length + 2 * value


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

    def __init__(self, platform, sequence, qubit):
        self.platform = platform
        self.sequence = sequence
        self.qubit = qubit

    def get(self):
        results = self.platform.execute_pulse_sequence(self.sequence)
        return list(list(results.values())[0].values())[0] #TODO: Replace with the particular acquisition
