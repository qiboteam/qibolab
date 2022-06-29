import win32com.client #imports the pywin32 library
from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
class Teledyine_scope(AbstractInstrument):
    """
    """

    def __init__(self, name, address) -> None:
        super().__init__(name, address)
        self.scope: win32com.client
    
    def start(self):
        pass
    def connect(self):
        try:   
            self.scope=win32com.client.Dispatch("LeCroy.ActiveDSOCtrl.1")  #creates instance of the ActiveDSO control
        except:
            raise SystemError("Need to run on Windows and to install 'activatedsoinstaller.exe'")
        self.scope.MakeConnection(f"IP:{self.address}") #Connects to the oscilloscope.  Substitute your IP address

    def setup(self):
        pass
    
    def disconnect(self):
        self.scope.Disconnect() #Disconnects from the oscilloscope
    def stop(self):
        pass
    
    def get_LargeUnscaledWaveform(self, chan):
        """
        Returns an unscaled waveform. This should be used with very large waveforms.
        """
        buffer = self.scope.GetByteWaveform(f"C{chan}", 5000, 0)
        data = []
        for buf in buffer:
            data += [buf]
        return np.array(data)

    def get_Waveform(self, chan):
        """
        """
        if isinstance(chan, list):
            arg = str(chan).replace("[", "C").replace("]", "").replace(" ","").replace(",", ", C")
            print(arg)
        else:
            arg = f"C{chan}"
        buffer = self.scope.GetScaledWaveform(arg, 5000, 0)
        data = []
        for buf in buffer:
            data += [buf]
        return np.array(data)

    def set_VoltPerDiv(self, chan, value):
        """
        Might be useless because the scope always read its full voltage range, and returns it
        with "get_Waveform". 
        """
        self.VoltPerDiv = value
        self.scope.WriteString(f"C{chan}:VDIV {value}",1) #Remote Command to set C1 volt/div setting to 20 mV.
    
        

scope = Teledyine_scope("Qcomp_scope", "192.168.0.30")
scope.connect()
scope.set_VoltPerDiv(2, 1)
data = scope.get_Waveform([1, 2])
print(data)
plt.figure()
plt.plot(data)
plt.show()