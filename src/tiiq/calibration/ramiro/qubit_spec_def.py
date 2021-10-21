#Set up the environment. 
import os 
import scipy.signal
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import lmfit
import xarray as xr

from qcodes import ManualParameter, Parameter

from pathlib import Path
from quantify.data.handling import get_datadir, set_datadir
from quantify.measurement import MeasurementControl
from quantify.measurement.control import Settable, Gettable
import quantify.visualization.pyqt_plotmon as pqm
from quantify.visualization.instrument_monitor import InstrumentMonitor
from qcodes.instrument import Instrument

import QRM_def as qrmd
import QCM_def as qcmd
import RohdeSchwarz_SGS100A_def as RS_SGS100A

class qubit_spec():

    def __init__(self,info):
        self.info = info

    def instantiate_instruments(self):
        info = self.info
        RS_LO_qcm = info["RohdeSchwar_LO_qcm"]
        RS_LO_qrm = info["RohdeSchwar_LO_qrm"]

        qrm = qrmd.Qblox_QRM(info)
        qcm = qcmd.Qblox_QCM(info)

        self.qrm = qrm
        self.qcm = qcm

        self.LO_qrm = RS_SGS100A.RohdeSchwarz_SGS100A(RS_LO_qrm)
        self.LO_qcm = RS_SGS100A.RohdeSchwarz_SGS100A(RS_LO_qcm)

        qrm.set_data_dictionary()
        qrm.connect_qrm_ip()
        qcm.connect_qcm_ip()
        qrm.reset_qrm()
        qcm.reset_qcm()

        qrm.reference_clock_external()

        self.LO_qrm.connect_RohdeSchwarz_SGS100A()
        self.LO_qcm.connect_RohdeSchwarz_SGS100A()

        self.LO_qrm.turn_on_RohdeSchwarz_SGS100A()
        self.LO_qcm.turn_on_RohdeSchwarz_SGS100A()


        self.MC = MeasurementControl('MC')
        self.plotmon = pqm.PlotMonitor_pyqt('plotmon')

        self.insmon = InstrumentMonitor("Instruments Monitor")
        # Create the live plotting intrument which handles the graphical interface
        # Two windows will be created, the main will feature 1D plots and any 2D plots will go to the secondary

        # Connect the live plotting monitor to the measurement control
        self.MC.instr_plotmon(self.plotmon.name)
        # The instrument monitor will give an overview of all parameters of all instruments

        # By connecting to the MC the parameters will be updated in real-time during an experiment.
        self.MC.instrument_monitor(self.insmon.name)       
        
        self.pars = Instrument('ParameterHolder')
        self.pars.add_parameter('qcm_leng', initial_value=0, unit='ns', label='Time', parameter_class=ManualParameter)
        self.pars.add_parameter('t1_wait', initial_value=0, unit='ns', label='Time', parameter_class=ManualParameter)
        self.pars.add_parameter('ramsey_wait', initial_value=0, unit='ns', label='Time', parameter_class=ManualParameter)

    def prepare_setup(self):
        info = self.info
        RS_LO_qcm = info["RohdeSchwar_LO_qcm"]
        RS_LO_qrm = info["RohdeSchwar_LO_qrm"]

        LO_qcm = self.LO_qcm
        LO_qrm = self.LO_qrm

        qrm = self.qrm
        qcm = self.qcm

        qrm.set_waveforms()
        qcm.set_waveforms()
        qrm.modulate_envolope()


        qrm.specify_acquisitions()
        qrm.sequence_program_qubit_spec(qcm.leng)
        qcm.sequence_program_qubit_spec()

        qrm.upload_waveforms()
        qcm.upload_waveforms(qrm.acquisitions)

        qrm.enable_hardware_averaging()
        qcm.set_sync_and_gain_of_sequencer()
        qrm.configure_sequencer_sync()
        print(qcm.pulsar_qcm.sequencer0_gain_awg_path0())
        qcm.pulsar_qcm.sequencer0_gain_awg_path0(self.info['waveform_qcm_1']['gain'])
        qcm.pulsar_qcm.sequencer0_gain_awg_path1(self.info['waveform_qcm_1']['gain'])
        print(qcm.pulsar_qcm.sequencer0_gain_awg_path0())
        LO_qcm.mw_LO.power(info['RohdeSchwar_LO_qcm']['power'])
        LO_qrm.mw_LO.power(info['RohdeSchwar_LO_qrm']['power'])
        LO_qcm.mw_LO.frequency(info['RohdeSchwar_LO_qcm']['frequency'])
        LO_qrm.mw_LO.frequency(info['RohdeSchwar_LO_qrm']['frequency'])
        LO_qcm.mw_LO.on()
        LO_qrm.mw_LO.on()

        MC = self.MC
        MC.soft_avg(info["software_averages"])

        wave_gettable = IQSignal_qubit_freq_spec(info,qrm,qcm)
        self.ro_gettable = Gettable(wave_gettable)

    def prepare_setup_rabi_length(self):
        info = self.info
        RS_LO_qcm = info["RohdeSchwar_LO_qcm"]
        RS_LO_qrm = info["RohdeSchwar_LO_qrm"]

        LO_qcm = self.LO_qcm
        LO_qrm = self.LO_qrm

        qrm = self.qrm
        qcm = self.qcm

        qrm.set_waveforms()
        qrm.modulate_envolope()


        qrm.specify_acquisitions()
    
        qcm.sequence_program_qubit_spec()

        qrm.enable_hardware_averaging()
        qcm.set_sync_and_gain_of_sequencer()
        qrm.configure_sequencer_sync()
        print(qcm.pulsar_qcm.sequencer0_gain_awg_path0())
        qcm.pulsar_qcm.sequencer0_gain_awg_path0(self.info['waveform_qcm_1']['gain'])
        qcm.pulsar_qcm.sequencer0_gain_awg_path1(self.info['waveform_qcm_1']['gain'])
        print(qcm.pulsar_qcm.sequencer0_gain_awg_path0())
        LO_qcm.mw_LO.power(info['RohdeSchwar_LO_qcm']['power'])
        LO_qrm.mw_LO.power(info['RohdeSchwar_LO_qrm']['power'])
        LO_qcm.mw_LO.frequency(info['RohdeSchwar_LO_qcm']['frequency'])
        LO_qrm.mw_LO.frequency(info['RohdeSchwar_LO_qrm']['frequency'])
        LO_qcm.mw_LO.on()
        LO_qrm.mw_LO.on()

        MC = self.MC
        MC.soft_avg(info["software_averages"])

        wave_gettable = IQSignal_qubit_freq_spec(info,qrm,qcm)
        self.ro_gettable = Gettable(wave_gettable)

    def prepare_setup_ramsey(self):
        info = self.info
        RS_LO_qcm = info["RohdeSchwar_LO_qcm"]
        RS_LO_qrm = info["RohdeSchwar_LO_qrm"]

        LO_qcm = self.LO_qcm
        LO_qrm = self.LO_qrm

        qrm = self.qrm
        qcm = self.qcm

        qrm.set_waveforms()
        qrm.modulate_envolope()
        qcm.set_waveforms()

        qrm.specify_acquisitions()

        qrm.enable_hardware_averaging()
        qcm.set_sync_and_gain_of_sequencer()
        qrm.configure_sequencer_sync()
        print(qcm.pulsar_qcm.sequencer0_gain_awg_path0())
        qcm.pulsar_qcm.sequencer0_gain_awg_path0(self.info['waveform_qcm_1']['gain'])
        qcm.pulsar_qcm.sequencer0_gain_awg_path1(self.info['waveform_qcm_1']['gain'])
        print(qcm.pulsar_qcm.sequencer0_gain_awg_path0())
        LO_qcm.mw_LO.power(info['RohdeSchwar_LO_qcm']['power'])
        LO_qrm.mw_LO.power(info['RohdeSchwar_LO_qrm']['power'])
        LO_qcm.mw_LO.frequency(info['RohdeSchwar_LO_qcm']['frequency'])
        LO_qrm.mw_LO.frequency(info['RohdeSchwar_LO_qrm']['frequency'])
        LO_qcm.mw_LO.on()
        LO_qrm.mw_LO.on()

        MC = self.MC
        MC.soft_avg(info["software_averages"])

        #wave_gettable = IQSignal_qubit_freq_specy(info,qrm,qcm)
        #self.ro_gettable = Gettable(wave_gettable)

    def run_qubit_freq_spec(self):
        qrm = self.qrm
        qcm = self.qcm
        self.prepare_setup()
        info = self.info
        RS_LO_qcm = info["RohdeSchwar_LO_qcm"]
        RS_LO_qrm = info["RohdeSchwar_LO_qrm"]
        MC = self.MC
        LO_qcm = self.LO_qcm
        LO_qrm = self.LO_qrm
        plotmon = self.plotmon
        insmon = self.insmon
        LO_qcm_freq_start = RS_LO_qcm["frequency_start"]
        LO_qcm_freq_stop = RS_LO_qcm["frequency_stop"]
        LO_qcm_freq_step = RS_LO_qcm["frequency_step"]

        #MC.settables(pars.leng)
        MC.settables(LO_qcm.mw_LO.frequency)
        # as a QCoDeS parameter, 't' obeys the JSON schema for a valid Settable and can be passed to the MC directly.
        #MC.setpoints(np.arange(leng_start,leng_stop,leng_step))
        MC.setpoints(np.arange(LO_qcm_freq_start,LO_qcm_freq_stop,LO_qcm_freq_step))
        MC.gettables(self.ro_gettable)                   
        # as a QCoDeS parameter, 'sig' obeys the JSON schema for a valid Gettable and can be passed to the MC directly.
        dataset = MC.run('QubitSpec')

        qrm.stop_sequencers()
        qcm.stop_sequencers()
        LO_qcm.turn_off_RohdeSchwarz_SGS100A()
        LO_qrm.turn_off_RohdeSchwarz_SGS100A()

    def run_rabi_gain(self,setpoints):
        self.prepare_setup()
        MC = self.MC
        plotmon = self.plotmon
        insmon = self.insmon
        qrm = self.qrm
        qcm = self.qcm
        LO_qrm = self.LO_qrm
        LO_qcm = self.LO_qcm

        #MC.settables(pars.leng)
        MC.settables(gain_parameter(awg=self.qcm.pulsar_qcm))
        # as a QCoDeS parameter, 't' obeys the JSON schema for a valid Settable and can be passed to the MC directly.
        #MC.setpoints(np.arange(leng_start,leng_stop,leng_step))
        MC.setpoints(setpoints)
        MC.gettables(self.ro_gettable)                   
        # as a QCoDeS parameter, 'sig' obeys the JSON schema for a valid Gettable and can be passed to the MC directly.
        dataset = MC.run('QubitSpec')

        qrm.stop_sequencers()
        qcm.stop_sequencers()
        LO_qcm.turn_off_RohdeSchwarz_SGS100A()
        LO_qrm.turn_off_RohdeSchwarz_SGS100A()

    def run_rabi_gain_freq(self,setpoints_gain,setpoints_freq):
        self.prepare_setup()
        MC = self.MC
        plotmon = self.plotmon
        insmon = self.insmon
        qrm = self.qrm
        qcm = self.qcm
        LO_qrm = self.LO_qrm
        LO_qcm = self.LO_qcm

        #MC.settables(pars.leng)
        MC.settables([gain_parameter(awg=self.qcm.pulsar_qcm),
                      LO_qcm.mw_LO.frequency])
        # as a QCoDeS parameter, 't' obeys the JSON schema for a valid Settable and can be passed to the MC directly.
        #MC.setpoints(np.arange(leng_start,leng_stop,leng_step))
        MC.setpoints_grid([setpoints_gain,setpoints_freq])
        MC.gettables(self.ro_gettable)                   
        # as a QCoDeS parameter, 'sig' obeys the JSON schema for a valid Gettable and can be passed to the MC directly.
        dataset = MC.run('QubitSpec')

        qrm.stop_sequencers()
        qcm.stop_sequencers()
        LO_qcm.turn_off_RohdeSchwarz_SGS100A()
        LO_qrm.turn_off_RohdeSchwarz_SGS100A()

    def run_rabi_length(self,setpoints_length):
        self.prepare_setup_rabi_length()
        MC = self.MC
        plotmon = self.plotmon
        insmon = self.insmon
        qrm = self.qrm
        qcm = self.qcm
        LO_qrm = self.LO_qrm
        LO_qcm = self.LO_qcm

        ro_gettable = IQSignal_rabi_waveform_leng(self.info,qrm,qcm,self.pars)
        self.ro_gettable = Gettable(ro_gettable)
        #MC.settables(pars.leng)
        MC.settables(self.pars.qcm_leng)
        # as a QCoDeS parameter, 't' obeys the JSON schema for a valid Settable and can be passed to the MC directly.
        #MC.setpoints(np.arange(leng_start,leng_stop,leng_step))
        MC.setpoints(setpoints_length)
        MC.gettables(self.ro_gettable)                   
        # as a QCoDeS parameter, 'sig' obeys the JSON schema for a valid Gettable and can be passed to the MC directly.
        dataset = MC.run('Rabi_Duration')

        qrm.stop_sequencers()
        qcm.stop_sequencers()
        LO_qcm.turn_off_RohdeSchwarz_SGS100A()
        LO_qrm.turn_off_RohdeSchwarz_SGS100A()

    def run_t1(self,setpoints_time):
        # NEEDS TO BE CHANGED AND DEBUGGED
        # prepare_setup
        self.prepare_setup_rabi_length()
        MC = self.MC
        plotmon = self.plotmon
        insmon = self.insmon
        qrm = self.qrm
        qcm = self.qcm
        LO_qrm = self.LO_qrm
        LO_qcm = self.LO_qcm

        qcm.set_waveforms()
        qcm.sequence_program_qubit_spec()
        qcm.upload_waveforms(qrm.acquisitions)

        ro_gettable = IQSignal_t1(self.info,qrm,qcm,self.pars)
        self.ro_gettable = Gettable(ro_gettable)
        #MC.settables(pars.leng)
        MC.settables(self.pars.t1_wait)
        # as a QCoDeS parameter, 't' obeys the JSON schema for a valid Settable and can be passed to the MC directly.
        #MC.setpoints(np.arange(leng_start,leng_stop,leng_step))
        MC.setpoints(setpoints_time)
        MC.gettables(self.ro_gettable)                   
        # as a QCoDeS parameter, 'sig' obeys the JSON schema for a valid Gettable and can be passed to the MC directly.
        dataset = MC.run(f'T1_{setpoints_time.max()}')

        qrm.stop_sequencers()
        qcm.stop_sequencers()
        LO_qcm.turn_off_RohdeSchwarz_SGS100A()
        LO_qrm.turn_off_RohdeSchwarz_SGS100A()

    def run_rabi_length_freq(self,setpoints_length,setpoints_freq):
        self.prepare_setup()
        MC = self.MC
        plotmon = self.plotmon
        insmon = self.insmon
        qrm = self.qrm
        qcm = self.qcm
        LO_qrm = self.LO_qrm
        LO_qcm = self.LO_qcm

        ro_gettable = IQSignal_rabi_waveform_leng(self.info,qrm,qcm,self.pars)
        self.ro_gettable = Gettable(ro_gettable)
        #MC.settables(pars.leng)
        MC.settables([self.pars.qcm_leng,
                      LO_qcm.mw_LO.frequency])
        # as a QCoDeS parameter, 't' obeys the JSON schema for a valid Settable and can be passed to the MC directly.
        #MC.setpoints(np.arange(leng_start,leng_stop,leng_step))
        MC.setpoints_grid([setpoints_length,setpoints_freq])
        MC.gettables(self.ro_gettable)                   
        # as a QCoDeS parameter, 'sig' obeys the JSON schema for a valid Gettable and can be passed to the MC directly.
        dataset = MC.run('Rabi_Freq_Duration')

        qrm.stop_sequencers()
        qcm.stop_sequencers()
        LO_qcm.turn_off_RohdeSchwarz_SGS100A()
        LO_qrm.turn_off_RohdeSchwarz_SGS100A()

    def run_rabi_length_gain(self,setpoints_length,setpoints_gain):
        self.prepare_setup_rabi_length()
        MC = self.MC
        plotmon = self.plotmon
        insmon = self.insmon
        qrm = self.qrm
        qcm = self.qcm
        LO_qrm = self.LO_qrm
        LO_qcm = self.LO_qcm

        ro_gettable = IQSignal_rabi_waveform_leng(self.info,qrm,qcm,self.pars)
        self.ro_gettable = Gettable(ro_gettable)
        #MC.settables(pars.leng)
        MC.settables([self.pars.qcm_leng,
                      gain_parameter(awg=self.qcm.pulsar_qcm)])
        # as a QCoDeS parameter, 't' obeys the JSON schema for a valid Settable and can be passed to the MC directly.
        #MC.setpoints(np.arange(leng_start,leng_stop,leng_step))
        MC.setpoints_grid([setpoints_length,setpoints_gain])
        MC.gettables(self.ro_gettable)                   
        # as a QCoDeS parameter, 'sig' obeys the JSON schema for a valid Gettable and can be passed to the MC directly.
        dataset = MC.run('Rabi_Freq_Gain')

        qrm.stop_sequencers()
        qcm.stop_sequencers()
        LO_qcm.turn_off_RohdeSchwarz_SGS100A()
        LO_qrm.turn_off_RohdeSchwarz_SGS100A()

    def run_resonator_scan(self,setpoints_freq,mmt_label='Resonator_Scan'):
        self.prepare_setup()
        MC = self.MC
        plotmon = self.plotmon
        insmon = self.insmon
        qrm = self.qrm
        qcm = self.qcm
        LO_qrm = self.LO_qrm
        LO_qcm = self.LO_qcm
        LO_qcm.mw_LO.off()

        #MC.settables(pars.leng)
        MC.settables(LO_qrm.mw_LO.frequency)
        # as a QCoDeS parameter, 't' obeys the JSON schema for a valid Settable and can be passed to the MC directly.
        #MC.setpoints(np.arange(leng_start,leng_stop,leng_step))
        MC.setpoints(setpoints_freq)
        MC.gettables(self.ro_gettable)                   
        # as a QCoDeS parameter, 'sig' obeys the JSON schema for a valid Gettable and can be passed to the MC directly.
        dataset = MC.run(mmt_label)

        qrm.stop_sequencers()
        qcm.stop_sequencers()
        LO_qcm.turn_off_RohdeSchwarz_SGS100A()
        LO_qrm.turn_off_RohdeSchwarz_SGS100A()        

    def run_ramsey(self,setpoints_time):
        self.prepare_setup_ramsey()
        MC = self.MC
        plotmon = self.plotmon
        insmon = self.insmon
        qrm = self.qrm
        qcm = self.qcm
        LO_qrm = self.LO_qrm
        LO_qcm = self.LO_qcm

        ro_gettable = IQSignal_ramsey(self.info,qrm,qcm,self.pars)
        self.ro_gettable = Gettable(ro_gettable)
        #MC.settables(pars.leng)
        MC.settables(self.pars.ramsey_wait)
        # as a QCoDeS parameter, 't' obeys the JSON schema for a valid Settable and can be passed to the MC directly.
        #MC.setpoints(np.arange(leng_start,leng_stop,leng_step))
        MC.setpoints(setpoints_time)
        MC.gettables(self.ro_gettable)                   
        # as a QCoDeS parameter, 'sig' obeys the JSON schema for a valid Gettable and can be passed to the MC directly.
        dataset = MC.run(f'ramsey_{setpoints_time.max()}')

        qrm.stop_sequencers()
        qcm.stop_sequencers()
        LO_qcm.turn_off_RohdeSchwarz_SGS100A()
        LO_qrm.turn_off_RohdeSchwarz_SGS100A()

    def run_echo(self,setpoints_time):
        self.prepare_setup_ramsey()
        MC = self.MC
        plotmon = self.plotmon
        insmon = self.insmon
        qrm = self.qrm
        qcm = self.qcm
        LO_qrm = self.LO_qrm
        LO_qcm = self.LO_qcm

        ro_gettable = IQSignal_echo(self.info,qrm,qcm,self.pars)
        self.ro_gettable = Gettable(ro_gettable)
        #MC.settables(pars.leng)
        MC.settables(self.pars.ramsey_wait)
        # as a QCoDeS parameter, 't' obeys the JSON schema for a valid Settable and can be passed to the MC directly.
        #MC.setpoints(np.arange(leng_start,leng_stop,leng_step))
        MC.setpoints(setpoints_time)
        MC.gettables(self.ro_gettable)                   
        # as a QCoDeS parameter, 'sig' obeys the JSON schema for a valid Gettable and can be passed to the MC directly.
        dataset = MC.run(f'ramsey_{setpoints_time.max()}')

        qrm.stop_sequencers()
        qcm.stop_sequencers()
        LO_qcm.turn_off_RohdeSchwarz_SGS100A()
        LO_qrm.turn_off_RohdeSchwarz_SGS100A()

class gain_parameter():
    def __init__(self,awg):

        self.label = 'AWG Gain'
        self.unit = '(V/V)'
        self.name = 'awg gain'
        self.awg = awg
        
    def get(self):
        return self.awg.sequencer0_gain_awg_path0()
        
    def set(self,value):
        self.awg.sequencer0_gain_awg_path0(value)
        self.awg.sequencer0_gain_awg_path1(value)

class IQSignal_qubit_freq_spec():

    def __init__(self,info,qrm,qcm):

        self.label = ['Amplitude', 'Phase','I','Q']
        self.unit = ['V', 'Radians','V','V']
        self.name = ['A', 'Phi','I','Q']
        self.qrm = qrm
        self.qcm = qcm

    def get(self):

        qrm = self.qrm
        qcm = self.qcm
        '''
        qcm.set_waveforms_for_MC(self.pars)
        qrm.set_waveforms()
        qrm.specify_acquisitions()
        qrm.sequence_program_rabi()
        qcm.sequence_program_rabi()
        qrm.upload_waveforms_from_qcm(qcm.waveforms)
        qcm.upload_waveforms(qrm.acquisitions)

        '''
        qrm.arm_and_start_sequencer()
        qcm.arm_and_start_sequencer()
        qcm.wait_sequencer_to_stop()

        qrm.acquisition()
        #qrm.plot_acquisitions()

        #print(qrm.single_acq["single"]["acquisition"]["scope"]["path0"]["avg_cnt"])
        #print(qrm.single_acq["single"]["acquisition"]["scope"]["path1"]["avg_cnt"])
        preconfigured_getsignal = lambda : qrm.demodulate_and_integrate()
        i,q = preconfigured_getsignal()
        return np.sqrt(i**2+q**2),np.arctan2(q,i),i,q

class IQSignal_rabi_waveform_leng(IQSignal_qubit_freq_spec):
    def __init__(self,info,qrm,qcm,pars):
        super().__init__(info,qrm,qcm)
        self.pars = pars

    def get(self):
        qrm = self.qrm
        qcm = self.qcm
        pars = self.pars

        #qrm.set_waveforms()
        qcm.set_waveforms_length_scan(pars)
        #qrm.modulate_envolope()
        qcm.modulate_envolope()

        #qrm.specify_acquisitions()
        qrm.sequence_program_qubit_spec(qcm.leng)
        #qcm.sequence_program_qubit_spec()


        qrm.upload_waveforms()
        qcm.upload_waveforms(qrm.acquisitions)

        #qrm.enable_hardware_averaging()
        #qcm.set_sync_and_gain_of_sequencer()
        #qrm.configure_sequencer_sync()

        return super().get()

class IQSignal_t1(IQSignal_qubit_freq_spec):
    def __init__(self,info,qrm,qcm,pars):
        super().__init__(info,qrm,qcm)
        self.pars = pars

    def get(self):
        qrm = self.qrm
        qcm = self.qcm
        pars = self.pars

        #qrm.set_waveforms()
        #qrm.modulate_envolope()
        #qcm.modulate_envolope()

        #qrm.specify_acquisitions()
        qrm.sequence_program_t1(qcm_leng=self.qcm.leng,wait_time_ns=self.pars.t1_wait())
        #qcm.sequence_program_qubit_spec()

        qrm.upload_waveforms()
        qcm.upload_waveforms(qrm.acquisitions)

        #qrm.enable_hardware_averaging()
        #qcm.set_sync_and_gain_of_sequencer()
        #qrm.configure_sequencer_sync()

        return super().get()

class IQSignal_ramsey(IQSignal_qubit_freq_spec):
    def __init__(self,info,qrm,qcm,pars):
        super().__init__(info,qrm,qcm)
        self.pars = pars

    def get(self):
        qrm = self.qrm
        qcm = self.qcm
        pars = self.pars

        #qrm.set_waveforms()
        #qrm.modulate_envolope()
        #qcm.modulate_envolope()

        #qrm.specify_acquisitions()
        qrm.sequence_program_ramsey(qcm_leng=self.qcm.leng,wait_time_ns=self.pars.ramsey_wait())
        qcm.sequence_program_ramsey(wait_time_ns=self.pars.ramsey_wait())


        qrm.upload_waveforms()
        qcm.upload_waveforms(qrm.acquisitions)

        #qrm.enable_hardware_averaging()
        #qcm.set_sync_and_gain_of_sequencer()
        #qrm.configure_sequencer_sync()

        return super().get()

class IQSignal_echo(IQSignal_qubit_freq_spec):
    def __init__(self,info,qrm,qcm,pars):
        super().__init__(info,qrm,qcm)
        self.pars = pars

    def get(self):
        qrm = self.qrm
        qcm = self.qcm
        pars = self.pars

        #qrm.set_waveforms()
        #qrm.modulate_envolope()
        qcm.modulate_envolope_echo()

        #qrm.specify_acquisitions()
        qrm.sequence_program_echo(qcm_leng=self.qcm.leng,wait_time_ns=self.pars.ramsey_wait())
        qcm.sequence_program_echo(wait_time_ns=self.pars.ramsey_wait())


        qrm.upload_waveforms()
        qcm.upload_waveforms(qrm.acquisitions)

        #qrm.enable_hardware_averaging()
        #qcm.set_sync_and_gain_of_sequencer()
        #qrm.configure_sequencer_sync()

        return super().get()
