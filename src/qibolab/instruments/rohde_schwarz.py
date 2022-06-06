"""
Class to interface with the local oscillator RohdeSchwarz SGS100A
"""
from qibo.config import raise_error
from qibolab.instruments.abstract import AbstractInstrument, InstrumentException

from quantify_core.measurement.control import Gettable, Settable

class SGS100A(AbstractInstrument):

    def __init__(self, name, address):
        super().__init__(name, address)
        self.device_parameters = {}
        self.settable_frequency = Settable(self.FrequencyParameter(self))

    rw_property_wrapper = lambda parameter: property(lambda self: self.device.get(parameter), lambda self,x: self.set_device_parameter(parameter,x))
    power = rw_property_wrapper('power')
    frequency = rw_property_wrapper('frequency')

    def connect(self):
        """
        Connects to the instrument using the IP address set in the runcard.
        """
        if not self.is_connected:
            import qcodes.instrument_drivers.rohde_schwarz.SGS100A as LO_SGS100A
            from pyvisa.errors import VisaIOError
            for attempt in range(3):
                try:
                    self.device = LO_SGS100A.RohdeSchwarz_SGS100A(self.name, f"TCPIP0::{self.address}::5025::SOCKET")
                    self.is_connected = True
                    break
                except KeyError as exc:
                    print(f"Unable to connect:\n{str(exc)}\nRetrying...")
                    self.name += '_' + str(attempt)
                except Exception as exc:
                    print(f"Unable to connect:\n{str(exc)}\nRetrying...")
            if not self.is_connected:
                raise InstrumentException(self, f'Unable to connect to {self.name}')
        else:
            raise_error(Exception,'There is an open connection to the instrument already')


    def set_device_parameter(self, parameter: str, value):
        if not(parameter in self.device_parameters and self.device_parameters[parameter] == value):
            if self.is_connected:
                if hasattr(self.device, parameter):
                    self.device.set(parameter, value)
                    self.device_parameters[parameter] = value 
                    # DEBUG: Parameter Setting Printing
                    # print(f"Setting {self.name} {parameter} = {value}")
                else:
                    raise_error(Exception, f'The instrument {self.name} does not have parameter {parameter}')
            else:
                raise_error(Exception,'There is no connection to the instrument  {self.name}')


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
            self.device.close()
            self.is_connected = False

    class FrequencyParameter():
        label = 'Frequency'
        unit = 'Hz'
        name = 'frequency'
        
        def __init__(self, outter_class_instance):
            self.outter_class_instance = outter_class_instance

        def set(self, value):
            self.outter_class_instance.frequency =  value