"""
Class to interface with the SPI Rack Qutech Delft
"""
from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qblox_instruments import SpiRack
from qibo.config import raise_error

class SPI(AbstractInstrument):
    
    def __init__(self, name, port):
        super().__init__(name, port)

    def connect(self):
        """
        Connects to the instrument using the IP address set in the runcard.
        """
        if not self.is_connected:
            for attempt in range(3):
                try:
                    self.device = SpiRack(self.name, self.address)
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

    def setup(self, **kwargs):
        #Init S4g and D5a modules in SPI mapped on runcard 
        if self.is_connected:
            #TODO: Check data format from yml
            self.s4g_modules = kwargs.pop('s4g_modules')
            self.d5a_modules = kwargs.pop('d5a_modules')
            self.hardware_avg = kwargs['hardware_avg']
            self.sampling_rate = kwargs['sampling_rate']
            self.repetition_duration = kwargs['repetition_duration']
            self.minimum_delay_between_instructions = kwargs['minimum_delay_between_instructions']
            for s4g_module in self.s4g_modules.values():
                if not s4g_module[1] in self.device.instrument_modules:
                    self.device.add_spi_module(s4g_module[0], "S4g", s4g_module[1])    
            for d5a_module in self.d5a_modules.values():
                if not d5a_module[1] in self.device.instrument_modules:
                    self.device.add_spi_module(d5a_module[0], "D5a", d5a_module[1])
        else:
            raise_error(Exception,'There is no connection to the instrument')

        self.device.set_dacs_zero()
        return    

    def set_S4g_DAC_current(self, flux_port, current_value):
        #TODO: Check access to module info from stored mapping data
        #TODO: Test eval function from string 
        module = self.s4g_modules[flux_port]
        funtion = "self.device."+module[1]+".dac"+module[2]+".current(current_value)"
        eval(funtion, {"current_value": current_value})
        #setattr(self.device, attr, current)
    
    def get_S4g_DAC_current(self, flux_port):
        module = self.s4g_modules[flux_port]
        funtion = "self.device."+module[1]+".dac"+module[2]+".current()"
        return eval(funtion)
    
    def set_S4g_span(self, flux_port, span_value):
        module = self.s4g_modules[flux_port]
        function = "self.device."+module[1]+".dac"+module[2]+".span(span_value)"
        eval(function, {"span_value": span_value})

    def get_S4g_span(self, flux_port):
        module = self.s4g_modules[flux_port]
        function = "self.device."+module[1]+".dac"+module[2]+".span()"
        return eval(function)

    def set_D5a_DAC_voltage(self, flux_port, voltage_value):
        module = self.d5a_modules[flux_port]
        function = "self.device."+module[1]+".dac"+module[2]+".voltage(voltage_value)"
        eval(function, {"volatge_value": voltage_value})

    def get_D5a_DAC_voltage(self, flux_port):
        module = self.d5a_modules[flux_port]
        function = "self.device."+module[1]+".dac"+module[2]+"voltage()"
        return eval(function)
    
    def set_D5a_span(self, flux_port, span_value):
        module = self.d5a_modules[flux_port]
        function = "self.device."+module[1]+".dac"+module[2]+".span(span_value)"
        eval(function, {"span_value": span_value})

    def get_D5a_span(self, flux_port):
        module = self.d5a_modules[flux_port]
        function = "self.device."+module[1]+".dac"+module[2]+".span()"
        return eval(function)

    def set_SPI_DACS_to_cero(self):
        self.device.set_dacs_zero()

    def get_SPI_IDN(self):
        return self.device.IDN()

    def get_SPI_temperature(self):
        return self.device.temperature()

    def get_SPI_battery_voltage(self):
        return self.device.battery_voltages()
    
    def __del__(self):
        self.disconnect()

    def disconnect(self):
        if self.is_connected:
            self.device.close()
            self.is_connected = False

    def close(self):
        if self.is_connected:
            self.device.close()
            self.is_connected = False

    def start(self):
        #set the dacs to the values stored for each qubit in the runcard
        if self.is_connected:
            for s4g_module in self.s4g_modules.values():
                funtion = "self.device."+s4g_module[1]+".dac"+s4g_module[2]+".current(current_value)"
                eval(funtion, {"current_value": s4g_module[3]})
            for d5a_module in self.d5a_modules.values():
                funtion = "self.device."+d5a_module[1]+".dac"+d5a_module[2]+".current(current_value)"
                eval(funtion, {"current_value": d5a_module[3]})

    def stop(self):
        if self.is_connected:
            self.device.set_dacs_zero()