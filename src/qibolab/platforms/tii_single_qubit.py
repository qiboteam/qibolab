import numpy as np
import json

from qibolab.instruments.rohde_schwarz import SGS100A
from qibolab.instruments.qblox import Pulsar_QCM
from qibolab.instruments.qblox import Pulsar_QRM

import qibolab.calibration.fitting

from quantify_core.data.handling import get_datadir, set_datadir
from quantify_core.measurement import MeasurementControl
from quantify_core.measurement.control import Settable, Gettable
from quantify_core.visualization.pyqt_plotmon import PlotMonitor_pyqt
from quantify_core.visualization.instrument_monitor import InstrumentMonitor


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


class TIISingleQubit():

    def __init__(self):
        self._LO_qrm = SGS100A("LO_qrm", '192.168.0.7')
        self._LO_qcm = SGS100A("LO_qcm", '192.168.0.101')
        self._qrm = Pulsar_QRM("qrm", '192.168.0.2')
        self._qcm = Pulsar_QCM("qcm", '192.168.0.3')

        self._MC = MeasurementControl('MC')
        self._plotmon = PlotMonitor_pyqt('Plot Monitor')
        self._insmon = InstrumentMonitor("Instruments Monitor")

        self._MC.instr_plotmon(self._plotmon.name)
        self._MC.instrument_monitor(self._insmon.name)

        self._settings = None
        self._QRM_settings = None
        self._QRM_init_settings = None
        self._QCM_settings = None
        self._QCM_init_settings = None
        self._LO_QRM_settings = None
        self._LO_QCM_settings = None

        self._config_filename = "tii_single_qubit_config.json"
        self.load_setting_from_file(self._config_filename)
        set_datadir(self._settings.get('data_folder'))

    def load_setting_from_file(self, filename):
        """Read platform settings from json file."""
        with open(filename, "r") as file:
            data = json.load(file)
        for name, value in data.items():
            if not hasattr(self, name):
                raise KeyError(f"Unknown argument {name} passed in config json.")
            if getattr(self, name) is not None:
                raise KeyError(f"Cannot set {name} from json as it is already set.")
            setattr(self, name, value)

    def setup(self):
        self._LO_qrm.setup(self._LO_QRM_settings)
        self._LO_qcm.setup(self._LO_QCM_settings)
        self._qrm.setup(self._QRM_settings)
        self._qcm.setup(self._QCM_settings)

    def stop(self):
        self._LO_qrm.off()
        self._LO_qcm.off()
        self._qrm.stop()
        self._qcm.stop()

    def close(self):
        self._LO_qrm.close()
        self._LO_qcm.close()
        self._qrm.close()
        self._qcm.close()

    def __del__(self):
        self.close()

    def run_resonator_spectroscopy(self):
        self.load_setting_from_file(self._config_filename)
        self.setup()
        scanrange = variable_resolution_scanrange(lowres_width= 30e6, lowres_step= 1e6, highres_width= 1e6, highres_step= 0.05e6)

        self._MC.settables(self._LO_qrm.LO.frequency)
        self._MC.setpoints(scanrange + self._LO_QRM_settings['frequency'])
        self._MC.gettables(Gettable(ROController(self._qrm, self._qcm)))
        self._LO_qrm.on()
        self._LO_qcm.off()
        dataset = self._MC.run('Resonator Spectroscopy', soft_avg = self._settings['software_averages'])
        # http://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html
        # print(dataset)
        # freq = fit_pulse(dataset.[''], freq_sweep)
        self.stop()
        return dataset

    def run_qubit_spectroscopy(self):
        self.load_setting_from_file(self._config_filename)
        self._qcm._settings['pulses']['qc_pulse']['length'] = 3000
        self._qrm._settings['pulses']['ro_pulse']['start'] = self._qcm._settings['pulses']['qc_pulse']['length']+40
        self.setup()
        scanrange = variable_resolution_scanrange(lowres_width= 30e6, lowres_step= 2e6, highres_width= 5e6, highres_step= 0.2e6)

        self._MC.settables(self._LO_qcm.LO.frequency)
        self._MC.setpoints(scanrange + self._LO_QCM_settings['frequency'])
        self._MC.gettables(Gettable(ROController(self._qrm, self._qcm)))
        self._LO_qrm.on()
        self._LO_qcm.on()
        dataset = self._MC.run('Resonator Spectroscopy', soft_avg = self._settings['software_averages'])
        self.stop()

    def run_Rabi_pulse_length(self):
        self.setup()
        self._MC.settables(QCPulseLengthParameter(self._qrm, self._qcm))
        self._MC.setpoints(np.arange(1,3000,5))
        self._MC.gettables(Gettable(ROController(self._qrm, self._qcm)))
        self._LO_qrm.on()
        self._LO_qcm.on()
        dataset = self._MC.run('Rabi Pulse Length', soft_avg = self._settings['software_averages'])
        self.stop()

    def run_Rabi_pulse_gain(self):
        pass
    def run_t1(self):
        pass
    def run_ramsey(self):
        pass
    def run_spin_echo(self):
        pass


class ROController():

    # Quantify Gettable Interface Implementation
    label = ['Amplitude', 'Phase','I','Q']
    unit = ['V', 'Radians','V','V']
    name = ['A', 'Phi','I','Q']

    def __init__(self, qrm: Pulsar_QRM, qcm: Pulsar_QCM):
        self._qrm = qrm
        self._qcm = qcm

    def get(self):
        qrm = self._qrm
        qcm = self._qcm

        qrm.setup(qrm._settings)
        qrm.set_waveforms_from_pulses_definition(qrm._settings['pulses'])
        qrm.set_program_from_parameters(qrm._settings)
        qrm.set_acquisitions()
        qrm.set_weights()
        qrm.upload_sequence()

        qcm.setup(qcm._settings)
        qcm.set_waveforms_from_pulses_definition(qcm._settings['pulses'])
        qcm.set_program_from_parameters(qcm._settings)
        qcm.set_acquisitions()
        qcm.set_weights()
        qcm.upload_sequence()

        qcm.play_sequence()
        acquisition_results = qrm.play_sequence_and_acquire()
        return acquisition_results

class QCPulseLengthParameter():

    label = 'Qubit Control Pulse Length'
    unit = 'ns'
    name = 'qc_pulse_length'

    def __init__(self, qrm: Pulsar_QRM, qcm: Pulsar_QCM):
        self._qrm = qrm
        self._qcm = qcm

    def set(self,value):
        self._qcm._settings['pulses']['qc_pulse']['length']=value
        self._qrm._settings['pulses']['ro_pulse']['start']=value+40

class QCPulseGainParameter():

    label = 'Qubit Control Gain'
    unit = '(V/V)'
    name = 'qc_pulse_gain'

    def __init__(self, qcm: Pulsar_QCM):
        self._qcm = qcm

    def set(self,value):
        sequencer = self._qcm._settings['sequencer']
        if sequencer == 1:
            self._qcm.sequencer1_gain_awg_path0(value)
            self._qcm.sequencer1_gain_awg_path1(value)
        else:
            self._qcm.sequencer0_gain_awg_path0(value)
            self._qcm.sequencer0_gain_awg_path1(value)


# T1: RX(pi) - wait t(rotates z) - readout
# Ramsey: RX(pi/2) - wait t(rotates z) - RX(pi/2) - readout
# Spin Echo: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - readout
# Spin Echo: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - RX(pi/2) - readout

# Ignore all functions after this point
# Check Ramiros code to generate sequences
    #def sequence_program_single(self):
    #def sequence_program_average(self):
    #def sequence_program_qubit_spec(self,qcm_leng,repetition_duration=200000):
    #def sequence_program_t1(self,qcm_leng,wait_time_ns=20,repetition_duration=200000):
    #def sequence_program_echo(self,qcm_leng,wait_time_ns=20,repetition_duration=200000):
    #def sequence_program_ramsey(self,qcm_leng,wait_time_ns=20,repetition_duration=200000):
