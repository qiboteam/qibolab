import numpy as np

from qibolab.instruments.rohde_schwarz import SGS100A
from qibolab.instruments.qblox import Pulsar_QCM
from qibolab.instruments.qblox import Pulsar_QRM

# from rohde_schwarz import SGS100A
# from qblox import Pulsar_QCM
# from qblox import Pulsar_QRM

from quantify_core.data.handling import get_datadir, set_datadir
from quantify_core.measurement import MeasurementControl
from quantify_core.measurement.control import Settable, Gettable
from quantify_core.visualization.pyqt_plotmon import PlotMonitor_pyqt
from quantify_core.visualization.instrument_monitor import InstrumentMonitor

class TIISingleQubit():

    _settings = {
        'data_dictionary': '.data/',
        "hardware_avg": 1024,
        "sampling_rate": 1e9,
        "software_averages": 1,
        "repetition_duration": 200000}
    _QRM_settings = {
            'gain': 0.4,
            'hardware_avg': _settings['hardware_avg'],
            'initial_delay': 0,
            "repetition_duration": 200000,
            'pulses': {
                'ro_pulse': {	"freq_if": 20e6,
                                "amplitude": 0.9,
                                "start": 300+40,
                                "length": 6000,
                                "offset_i": 0,
                                "offset_q": 0,
                                "shape": "Block",
                            }
                        },

            'start_sample': 130,
            'integration_length': 2500,
            'sampling_rate': _settings['sampling_rate'],
            'mode': 'ssb'}
    _QCM_settings = {
            'gain': 0.5,
            'hardware_avg': _settings['hardware_avg'],
            'initial_delay': 0,
            "repetition_duration": 200000,
            'pulses': {
                'qc_pulse':{	"freq_if": 200e6,
                            "amplitude": 0.25,
                            "length": 300,
                            "offset_i": 0,
                            "offset_q": 0,
                            "shape": "Gaussian",
                            "delay_before": 10, # cannot be 0
                            }
                        }}
    _LO_QRM_settings = { "power": 15,
                        "frequency":7.79813e9 - _QRM_settings['pulses']['ro_pulse']['freq_if']}
    _LO_QCM_settings = { "power": 12,
                        "frequency":8.724e9 + _QCM_settings['pulses']['qc_pulse']['freq_if']}

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

        set_datadir(self._settings['data_dictionary'])

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

    def run_resonator_spectroscopy(self):
        self.setup()
        lowres_width = 30e6
        lowres_step = 2e6
        highres_width = 0.5e6
        highres_step = 0.05e6
        scanrange = np.concatenate((np.arange(-lowres_width,-highres_width,lowres_step),np.arange(-highres_width,highres_width,highres_step),np.arange(highres_width,lowres_width,lowres_step)))

        self._MC.settables(self._LO_qrm.LO.frequency)
        self._MC.setpoints(scanrange + self._LO_QRM_settings['frequency'])
        self._MC.gettables(Gettable(ROController(self._qrm, self._qcm)))
        self._LO_qrm.on()
        self._LO_qcm.off()
        dataset = self._MC.run('Resonator Spectroscopy', soft_avg = self._settings['software_averages'])
        self.stop()

    def run_qubit_spectroscopy(self):
        self.setup()
        lowres_width = 30e6
        lowres_step = 2e6
        highres_width = 5e6
        highres_step = 0.2e6
        scanrange = np.concatenate((np.arange(-lowres_width,-highres_width,lowres_step),np.arange(-highres_width,highres_width,highres_step),np.arange(highres_width,lowres_width,lowres_step)))

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

        # qcm.play_sequence() # if sync enabled I believe it is not necessary
        return qrm.play_sequence_and_acquire()

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
        sequencer = self._device._settings['sequencer']
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

def sequence_program_single(self):
    seq_prog = """
    play    0,1,4     # Play waveforms (0,1) in channels (O0,O1) and wait 4ns.
    acquire 0,0,16380 # Acquire waveforms over remaining duration of acquisition of input vector of length = 16380 with integration weights 0,0
    stop              # Stop.
    """
    self.seq_prog = seq_prog

def sequence_program_average(self):
    seq_prog = f"""
        move    {self.info["number_of_average"]},R0
        nop
    loop:
        play    0,1,4
        acquire 0,0,16380
        loop    R0,@loop

        stop
    """
    self.seq_prog = seq_prog

def sequence_program_qubit_spec(self,qcm_leng,repetition_duration=200000):
    wait_loop_step=1000
    num_wait_loops,extra_wait = calculate_repetition_rate(repetition_duration=repetition_duration,
                                                            wait_loop_step=wait_loop_step,
                                                            duration_base=16384)
    buffer_time = 40  #ns


    seq_prog = f"""
        move    {self.info["number_of_average"]},R0
        nop
        wait_sync 4           # Synchronize sequencers over multiple instruments

    loop:
        wait      {qcm_leng+buffer_time} # idle for xx ns gaussian pulse + 40 ns buffer
        play      0,1,4      # Play waveforms (0,1) in channels (O0,O1) and wait 4ns.
        acquire   0,0,4      # Acquire waveforms over remaining duration of acquisition of input vector of length = 16380 with integration weights 0,0
        wait      {16380-4-qcm_leng-buffer_time}
        move      {num_wait_loops},R1      # repetion rate loop iterator
        nop
        reprateloop:
            wait      {wait_loop_step}
            loop      R1,@reprateloop
        wait      {extra_wait}
        loop    R0,@loop

        stop
    """


    self.seq_prog = seq_prog


def sequence_program_t1(self,qcm_leng,wait_time_ns=20,repetition_duration=200000):
    wait_loop_step=1000
    num_wait_loops,extra_wait = calculate_repetition_rate(repetition_duration=repetition_duration-wait_time_ns,
                                                            wait_loop_step=wait_loop_step,
                                                            duration_base=16384)
    buffer_time = 40  #ns


    seq_prog = f"""
        move    {self.info["number_of_average"]},R0
        nop
        wait_sync 4           # Synchronize sequencers over multiple instruments

    loop:
        wait      {qcm_leng+buffer_time} # idle for xx ns gaussian pulse + 40 ns buffer
        wait      {wait_time_ns}
        play      0,1,4      # Play waveforms (0,1) in channels (O0,O1) and wait 4ns.
        acquire   0,0,4      # Acquire waveforms over remaining duration of acquisition of input vector of length = 16380 with integration weights 0,0
        wait      {16380-4-qcm_leng-buffer_time}
        move      {num_wait_loops},R1      # repetion rate loop iterator
        nop
        reprateloop:
            wait      {wait_loop_step}
            loop      R1,@reprateloop
        wait      {extra_wait}
        loop    R0,@loop

        stop
    """


    self.seq_prog = seq_prog

def sequence_program_echo(self,qcm_leng,wait_time_ns=20,repetition_duration=200000):
    wait_loop_step=1000
    num_wait_loops,extra_wait = calculate_repetition_rate(repetition_duration=repetition_duration-wait_time_ns,
                                                            wait_loop_step=wait_loop_step,
                                                            duration_base=16384)
    buffer_time = 40  #ns




    seq_prog = f"""
        move    {self.info["number_of_average"]},R0
        nop
        wait_sync 4           # Synchronize sequencers over multiple instruments

    loop:
        wait      {3*qcm_leng+wait_time_ns+2*qcm_leng+buffer_time} # idle for xx ns gaussian pulse + 40 ns buffer
        play      0,1,4      # Play waveforms (0,1) in channels (O0,O1) and wait 4ns.
        acquire   0,0,4      # Acquire waveforms over remaining duration of acquisition of input vector of length = 16380 with integration weights 0,0
        wait      {16384-4-4-3*qcm_leng-2*qcm_leng-buffer_time}
        move      {num_wait_loops},R1      # repetion rate loop iterator
        nop
        reprateloop:
            wait      {wait_loop_step}
            loop      R1,@reprateloop
        wait      {extra_wait}
        loop    R0,@loop

        stop
    """

    self.seq_prog = seq_prog

def sequence_program_ramsey(self,qcm_leng,wait_time_ns=20,repetition_duration=200000):
    wait_loop_step=1000
    num_wait_loops,extra_wait = calculate_repetition_rate(repetition_duration=repetition_duration-wait_time_ns,
                                                            wait_loop_step=wait_loop_step,
                                                            duration_base=16384)
    buffer_time = 40  #ns


    seq_prog = f"""
        move    {self.info["number_of_average"]},R0
        nop
        wait_sync 4           # Synchronize sequencers over multiple instruments

    loop:
        wait      {qcm_leng*3+wait_time_ns+buffer_time} # idle for xx ns gaussian pulse + 40 ns buffer
        play      0,1,4      # Play waveforms (0,1) in channels (O0,O1) and wait 4ns.
        acquire   0,0,4      # Acquire waveforms over remaining duration of acquisition of input vector of length = 16380 with integration weights 0,0
        wait      {16384-4-4-3*qcm_leng-buffer_time}
        move      {num_wait_loops},R1      # repetion rate loop iterator
        nop
        reprateloop:
            wait      {wait_loop_step}
            loop      R1,@reprateloop
        wait      {extra_wait}
        loop    R0,@loop

        stop
    """
