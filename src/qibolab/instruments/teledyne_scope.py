import win32com.client #imports the pywin32 library
from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from lecroydso import LeCroyDSO, ActiveDSO
from qibo.config import raise_error, log

class Teledyine_scope(AbstractInstrument, LeCroyDSO):
    """
    Requirements:
        - Windows
        - Install activedsoinstaller.exe (currently in the instrument directory)

    Driver for Teledyne Scope. Python driver was already implemented by LeCroy and the list of functions available can be found on:
    https://lecroydso.readthedocs.io/en/latest/api/lecroydso.html

    All functions are pretty self-explanatory, and they are all inherited. 
    Two key functions were added here because it was not implemented on their driver. 

    The channels are referred as "C1", "C2" ... and SI units are used. 
    """

    def __init__(self, name, address) -> None:
        AbstractInstrument.__init__(self,name, address)
    
    def connect(self):
        if not self.is_connected:
            try:
                LeCroyDSO.__init__(self, ActiveDSO(f"IP:{self.address}"))
            except Exception as exc:
                    raise InstrumentException(self, str(exc))
            self._is_connected = True
        else: 
            raise_error(Exception,'There is an open connection to the instrument already')

    def start(self):
        pass

    def setup(self):
        pass
    
    def disconnect(self):
        pass
    def stop(self):
        pass
    
    def get_timedwaveform(self, chan):
        """
        Get the waveform with the time axis and correct voltage.
        Return: Time (s), Voltages (V)
        """
        self.validate_source(chan)        
        buffer = self._conn.aDSO.GetScaledWaveformWithTimes(chan, int(self.get_num_points()), 0)
        return np.array([buffer[0], buffer[1]])

    def get_scaledwaveform(self, chan):
        """
        Get the waveform with the correct voltage.
        Return: Voltages (V)
        """
        self.validate_source(chan)        
        buffer = self._conn.aDSO.GetScaledWaveform(chan, int(self.get_num_points()), 0)
        data = []
        for buf in buffer:
            data += [buf]
        return np.array(data)
    
        

