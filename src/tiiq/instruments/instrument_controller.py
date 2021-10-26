"""
CLASS FILE FOR INSTRUMENT COMMUNICATION AND UTILITY
"""

#import qblox, rhode_schwarz classes
from tiiq.instruments.rohde_schwarz import SGS100A
from tiiq.instruments.qblox import Pulsar_QCM
from tiiq.instruments.qblox import Pulsar_QRM

class InstrumentController():
    """
    InstrumentController class to interface and provide shortcut to instrument functions
    """
    def __init__():
        #instanciate and connect all setup hardware: AWG, QCM y QRM (parte de instantiate_instruments de Ramiro)
        self.LO_qrm = SGS100A("LO_qrm", '192.168.0.7')
        self.LO_qcm = SGS100A("LO_qcm", '192.168.0.8')
        self.qrm = Pulsar_QRM("qrm", '192.168.0.2')
        self.qcm = Pulsar_QCM("qcm", '192.168.0.3')

    def setup(self, LO_qcm_freq, LO_qcm_power, LO_qrm_freq, LO_qrm_power, QRM_settings: dict, QCM_settings: dict):
        #Pass from experiment or callibration class dictionaries characterizing the HW setup

        #setting up LO for QRM (resonator) and QCM (qubit)
        self.LO_qrm.set_power(LO_qrm_power)
        self.LO_qrm.set_frequency(LO_qrm_freq)
        self.LO_qcm.set_power(LO_qcm_power)
        self.LO_qcm.set_frequency(LO_qcm_freq)

        #setting up and configuring QRM parameters
        #Decidir si los parametros de integracion y modulacion deben ir aqui o en la genracion de la secuencia!!!!
        """
        Integration and modulation dict example for QRM
        QRM_info =
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
        self.qrm.setup(QRM_settings) #prepare QRM settings for upload waveforms and play sequence
        #self.qcm.setup(QCM_hw_settings) #prepare QRM settings for upload waveforms and play sequence

    def upload_waveform_sequence(self, QRM_wave_params: dict, QCM_wave_params: dict, sequence):
        """
        set waveforms, upload and load in Qblox awg QRM and QCM (comprobar todos los prepare_setup de Ramiro's code para ver diferencias)
        """
        """
        QRM parameters for charaterization of the waveform
        QRM_wave_params =
        {
            "IF_frequency" = 20e6
            "waveform_length" = 500 #nanoseconds
            "offset_i" = -0
            "offset_q" = -0
            "amplitude" = 0.5
        }
        """
        #characterize QRM waveform
        self.qrm.set_waveforms(QRM_wave_params[IF_frequency],
                               QRM_wave_params[waveform_length],
                               QRM_wave_params[offset_i],
                               QRM_wave_params[offset_q],
                               QRM_wave_params[amplitude])
        #characterize QCM waveform
        #"""
        #QCM parameters for charaterization of the waveform
        #QCM_wave_params =
        #{
            #"waveform_type": "Block", #"Gaussian" or "Block",
            #"amplitude": 0.5,
            #"amplitude_start": 0,
            #"amplitude_stop": 0.7,
            #"amplitude_step": 0.002,
            #"IF_frequency": 100e6, #it means the RF tone will be generated below
            #"waveform_length": 200, #nanoseconds
            #"waveform_length_start": 50, #nanoseconds"
            #"waveform_length_stop": 1000, #nanoseconds",
            #"waveform_length_step": 50, #nanoseconds,
            #"offset_i": -0,
            #"offset_q": -0,
            #"gain": 1.,
            #"standard_deviation": 50, #For Gaussian waveform",
        #}
        #"""
        #self.qcm.set_waveforms()

        #modulate QRM signal (UP conversion)
        self.qrm.modulate_envolope()

        #specify acquisitions params in QRM
        self.qrm.specify_acquisitions()

        #Load experiment sequence
        #sequence = Q1asm string?
        self.qrm.set_sequence(sequence)

        #upload waveforms to QRM
        self.qrm.upload_waveforms()

        #specify QRM HW averaging
        self.qrm.enable_hardware_averaging()

        #enable sequencer sync
        self.qrm.configure_sequencer_sync()


    def play(self):
        #prepare objects (MC, plotmon...) for plotting output
        #self.soft_avg_numnber = info["software_averages"] (number of software averages done by MC object)
        #run the experiment tru MC.run (see Ramiro's code run_experiments methods. Ex: run_resonator_scan)

    def read_out(self):
        #demodulate and integrate QRM signal


    def stop(self):
        #stop ALL instruments
        self.LO_qrm.off()
        self.LO_qcm.off()
        self.qrm.stop()
        #self.qcm.stop()
        logger.info("All instruments stopped")


    def exit(self):
        #close connection with ALL instruments
        self.qrm.close()
        #self.qrm.close()
        logger.info("All instrument connections closed")
