"""
Class to interface with the local oscillator RohdeSchwarz SGS100A
"""
from qibo.config import raise_error
from qibolab.instruments.abstract import AbstractInstrument, InstrumentException

class SGS100A(AbstractInstrument):

    def __init__(self, name, ip):
        super().__init__(name, ip)

    rw_property_wrapper = lambda parameter: property(lambda self: self.device.get(parameter), lambda self,x: self.device.set(parameter,x))
    power = rw_property_wrapper('power')
    frequency = rw_property_wrapper('frequency')

    def connect(self):
        if not self.is_connected:
            import qcodes.instrument_drivers.rohde_schwarz.SGS100A as LO_SGS100A
            try:
                self.device = LO_SGS100A.RohdeSchwarz_SGS100A(self.name, f"TCPIP0::{self.ip}::inst0::INSTR")
            except Exception as exc:
                raise InstrumentException(self, str(exc))
            self.is_connected = True
        else:
            raise_error(Exception,'There is an open connection to the instrument already')


    def setup(self, **kwargs):
        if self.is_connected:
            self.power = kwargs.pop('power')
            self.frequency = kwargs.pop('frequency')
            self.__dict__.update(kwargs)
        else:
            raise_error(Exception,'There is no connection to the instrument')

    def start(self):
        self.device.on()

    def stop(self):
        self.device.off()

    def disconnect(self):
        if self.is_connected:
            self.device.off()
            self.device.close()
            self.is_connected = False

    def __del__(self):
        self.disconnect()


    def on(self):
        self.device.on()

    def off(self):
        self.device.off()

    def close(self):
        if self.is_connected:
            self.off()
            self.device.close()
            self.is_connected = False


