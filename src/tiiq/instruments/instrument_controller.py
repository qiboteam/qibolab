"""
CLASS FILE FOR INSTRUMENT COMMUNICATION AND UTILITY
"""

#import qblox, rhode_schwarz and AcquisitionController classes
from rohde_schwarz import SGS100A
from qblox import Pulsar_QCM
from qblox import Pulsar_QRM
from acquisition_controller import AcquisitionController

import logging
logger = logging.getLogger(__name__)

class InstrumentController():
    """
    InstrumentController class to interface and provide shortcut to instrument functions
    """
    def __init__(self):
        #instanciate and connect all setup hardware: AWG, QCM y QRM (parte de instantiate_instruments de Ramiro)
        self.LO_qrm = SGS100A("LO_qrm", '192.168.0.7')
        self.LO_qcm = SGS100A("LO_qcm", '192.168.0.8')
        self.qrm = Pulsar_QRM("qrm", '192.168.0.2')
        self.qcm = Pulsar_QCM("qcm", '192.168.0.3')

        # Create the live plotting intrument which handles the graphical interface
        # Two windows will be created, the main will feature 1D plots and any 2D plots will go to the secondary
        self.ac = AcquisitionController('MC')

    def setup(self, LO_qcm_freq, LO_qcm_power, LO_qrm_freq, LO_qrm_power, QRM_settings: dict):
        #Pass from experiment or callibration class parameters characterizing the HW setup

        #setting up LO for QRM (resonator) and QCM (qubit)
        self.LO_qrm.set_power(LO_qrm_power)
        self.LO_qrm.set_frequency(LO_qrm_freq)
        self.LO_qcm.set_power(LO_qcm_power)
        self.LO_qcm.set_frequency(LO_qcm_freq)

        #setting up acquisition_controller for real time plotting and control
        self.ac.setup()

        #setting up QRM parameters
        """
        QRM readOut Integration and demodulation parameters
        Dict example for QRM
        QRM_settings =
        {
            "data_dictionary": "quantify-data/",
            "ref_clock": external,
            "start_sample": 130,
            'hardware_avg': 1024,
            "integration_length": 600,
            "sampling_rate": 1e9,
            "mode": "ssb"
        }
        """
        #prepare QRM settings for upload waveforms and play sequence
        self.qrm.setup(QRM_settings)

        #prepare QCM with the same clock reference than QRM
        self.qcm.setup(QRM_settings['ref_clock'])

    def upload_waveform_sequence(self, QRM_wave_params: dict, QCM_wave_params: dict, QRM_sequence, QCM_sequence):
        """
        set waveform parameters, upload and load waveform in Qblox awg QRM and QCM
        (comprobar todos los prepare_setup de Ramiro's code para ver diferencias)

        QRM parameters for charaterization of the waveform
        QRM_wave_params =
        {
            "IF_frequency" = 20e6
            "waveform_length" = 500 #nanoseconds
            "offset_i" = -0
            "offset_q" = -0
            "amplitude" = 0.5
            (waveform type and gain not needed?)
        }
        """
        #characterize QRM waveform
        self.qrm.set_waveforms(QRM_wave_params['IF_frequency'],
                               QRM_wave_params['waveform_length'],
                               QRM_wave_params['offset_i'],
                               QRM_wave_params['offset_q'],
                               QRM_wave_params['amplitude'])

        """
        QCM parameters for charaterization of the waveform
        QCM_wave_params =
        {
            "waveform_type": "Block", #"Gaussian" or "Block",
            "amplitude": 0.5,
            "IF_frequency": 100e6, #it means the RF tone will be generated below
            "waveform_length": 200, #nanoseconds
            "waveform_length_start": 50, #nanoseconds", (NOT USED IN RANIRO'S CODE)
            "waveform_length_stop": 1000, #nanoseconds", (NOT USED IN RANIRO'S CODE)
            "waveform_length_step": 50, #nanoseconds, (NOT USED IN RANIRO'S CODE)
            "offset_i": -0,
            "offset_q": -0,
            "gain": 1.,
            "standard_deviation": 50, #For Gaussian waveform", (NOT USED IN RANIRO'S CODE)
        }
        """
        #characterize QCM waveform
        self.qcm.set_waveforms(QCM_wave_params['waveform_type'],
                               QCM_wave_params['amplitude'],
                               QCM_wave_params['IF_frequency'],
                               QCM_wave_params['waveform_length'],
                               #QCM_wave_params['waveform_length_start'], #not used in Ramiro's code
                               #QCM_wave_params['waveform_length_stop'], #not used in Ramiro's code
                               #QCM_wave_params['waveform_length_step'], #not used in Ramiro's code
                               QCM_wave_params['offset_i'],
                               QCM_wave_params['offset_q'],
                               QCM_wave_params['gain'],
                               #QCM_wave_params['standard_deviation'] #calculated directly from waveform_length param
                              )

        #modulate QRM signal (UP conversion)
        self.qrm.modulate_envolope()
        self.qcm.modulate_envolope() #done hard coded in in Ramiro's code: qcm.set_waveforms()

        #Load experiment sequence
        #sequence = Q1asm string -- create the sequence to be played in your experiment/callibration code
        #Example:
        #seq_prog = """
            #play    0,1,4     # Play waveforms (0,1) in channels (O0,O1) and wait 4ns.
            #acquire 0,0,16380 # Acquire waveforms over remaining duration of acquisition of input vector of length = 16380 with integration weights 0,0
            #stop              # Stop.
        #"""
        self.qrm.set_sequence(QRM_sequence)
        self.qcm.set_sequence(QCM_sequence)

        #upload waveforms to QRM
        self.qrm.upload_waveforms()
        self.qcm.upload_waveforms(self.qrm.acquisitions)

        #set QCM gain. Gain defined by the wavweform uploaded
        self.qcm.set_gain(self.qcm.gain)

    def play(self):
        #turn on configured LOs
        self.LO_qrm.on()
        self.LO_qcm.on()

        #Play waveform sequences loaded in QRM and QCM (see Ramiros's code for functions like run_ramsey and private classes like IQSignal_)
        #ro_gettable = IQSignal_ramsey(self.info,qrm,qcm,self.pars)
        #self.ro_gettable = Gettable(ro_gettable)

        #Also plot in real time the experiment using AcquisitionController (MeasurementControl object)
        #MC.settables(self.pars.ramsey_wait)

        # as a QCoDeS parameter, 't' obeys the JSON schema for a valid Settable and can be passed to the MC directly.
        #MC.setpoints(np.arange(leng_start,leng_stop,leng_step))

        #MC.setpoints(setpoints_time)
        #MC.gettables(self.ro_gettable)

        # as a QCoDeS parameter, 'sig' obeys the JSON schema for a valid Gettable and can be passed to the MC directly.
        #dataset = MC.run(f'ramsey_{setpoints_time.max()}', soft_avg = self.soft_avg_numnber)


    def stop(self):
        #stop ALL instruments
        self.LO_qrm.off()
        self.LO_qcm.off()
        self.qrm.stop()
        self.qcm.stop()
        logger.info("All instruments stopped")

    def exit(self):
        #close connection with ALL instruments
        self.qrm.close()
        self.qcm.close()
        logger.info("All instrument connections closed")
