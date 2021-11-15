"""
CLASS FILE FOR INSTRUMENT COMMUNICATION AND UTILITY
"""

#import qblox, rhode_schwarz and AcquisitionController classes
from rohde_schwarz import SGS100A
from qblox import Pulsar_QCM
from qblox import Pulsar_QRM
from acquisition_controller import AcquisitionController
from quantify_core.data.handling import get_datadir, set_datadir
from quantify_core.measurement.control import Settable, Gettable

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
        #self.ac = AcquisitionController()
        self.MC = MeasurementControl('MC')
        self.plotmon = pqm.PlotMonitor_pyqt('plotmon')
        self.insmon = InstrumentMonitor("Instruments Monitor")

    def setup(self, LO_qrm_freq, LO_qrm_power, LO_qcm_freq, LO_qcm_power, QRM_settings: dict, QCM_settings: dict):
        #Pass from experiment or callibration class parameters characterizing the HW setup

        # Connect the live plotting monitor to the measurement control
        self.MC.instr_plotmon(self.plotmon.name)

        # The instrument monitor will give an overview of all parameters of all instruments
        # By connecting to the MC the parameters will be updated in real-time during an experiment.
        self.MC.instrument_monitor(self.insmon.name)

        #setting up LO for QRM (resonator) and QCM (qubit)
        self.LO_qrm.set_power(LO_qrm_power)
        self.LO_qrm.set_frequency(LO_qrm_freq)
        self.LO_qcm.set_power(LO_qcm_power)
        self.LO_qcm.set_frequency(LO_qcm_freq)

        #setting up QRM parameters
        self.qrm.setup(QRM_settings)

        #prepare QCM with the same clock reference than QRM
        self.qcm.setup(QCM_settings)

        #set folder for storage data
        set_datadir(QRM_settings['data_dictionary'])
        print(f"Data will be saved in:\n{get_datadir()}")


    def play(self, label, setpoints, soft_avg_numnber, **Kwargs):
        #Activate QRM Local oscillators
        self.LO_qrm.on()
        self.LO_qcm.on()

        for key, value in kwargs.items():
            if (key == frequency) self.MC.settables(self.LO_qcm.LO.frequency)

        self.MC.setpoints(setpoints)
        self.MC.gettables(Gettable(self.qrm)) #won't work in a plattform with QRM y QCM configuration
        dataset = MC.run(label, soft_avg = soft_avg_numnber)

    def stop(self):
        #stop ALL instruments
        self.LO_qrm.off()
        self.LO_qcm.off()
        self.qrm.stop()
        self.qcm.stop()
        self.qrm.close()
        self.qcm.close()
        logger.info("All instruments stopped and closed")
