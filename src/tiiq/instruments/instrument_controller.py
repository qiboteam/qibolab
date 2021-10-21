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
        self.qcm = Pulsar_QCM("qcm", '192.168.0.2')

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


        #self.qcm.setup(QCM_hw_settings) #???

    def generate_sequence(QRM_waveform_parameters: dict, QCM_waveform_parameters: dict):
        #set_waveforms
        #upload waveforms
        #define QRM/QCM sequence for experiment or callibration
        #play sequence
        #return/show results

    def stop():
        #stop ALL instruments
        self.LO_qrm.off()
        self.LO_qcm.off()
        self.qrm.stop()
        #self.qcm.stop()
        logger.info("All instruments stopped")


    def exit():
        #close connection with ALL instruments
        self.qrm.close()
        #self.qrm.close()
        logger.info("All instrument connections closed")
