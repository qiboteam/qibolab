"""
CLASS FILE FOR READOUT EXPERIMENTS DATA
"""

from qcodes import ManualParameter, Parameter
from pathlib import Path
from quantify.data.handling import get_datadir, set_datadir
from quantify.measurement import MeasurementControl
from quantify.measurement.control import Settable, Gettable
import quantify.visualization.pyqt_plotmon as pqm
from quantify.visualization.instrument_monitor import InstrumentMonitor
from qcodes.instrument import Instrument

class AcquisitionController():

    """
    AcquisitionController class to provide shortcut to readOut instruments
    """
    def __init__(self, label):
        #Instantiate Mesurement Control Object for ReadOut in real time
        self.MC = MeasurementControl(label)
        self.plotmon = pqm.PlotMonitor_pyqt('plotmon')
        self.insmon = InstrumentMonitor("Instruments Monitor")


    def setup(self):
        # Connect the live plotting monitor to the measurement control
        self.MC.instr_plotmon(self.plotmon.name)

        # The instrument monitor will give an overview of all parameters of all instruments
        # By connecting to the MC the parameters will be updated in real-time during an experiment.
        self.MC.instrument_monitor(self.insmon.name)

        self.pars = Instrument('ParameterHolder')
        self.pars.add_parameter('qcm_leng', initial_value=0, unit='ns', label='Time', parameter_class=ManualParameter)
        self.pars.add_parameter('t1_wait', initial_value=0, unit='ns', label='Time', parameter_class=ManualParameter)
        self.pars.add_parameter('ramsey_wait', initial_value=0, unit='ns', label='Time', parameter_class=ManualParameter)
