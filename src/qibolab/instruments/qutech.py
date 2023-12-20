"""Class to interface with the SPI Rack Qutech Delft."""
from qblox_instruments import SpiRack
from qibo.config import log, raise_error

from qibolab.instruments.abstract import Instrument, InstrumentException


class SPI(Instrument):
    property_wrapper = lambda parent, device, *parameter: property(
        lambda self: device.get(parameter[0]),
        lambda self, x: parent._set_device_parameter(device, *parameter, value=x),
    )

    def __init__(self, name, address):
        super().__init__(name, address)
        self.device: SpiRack = None
        self.s4g_modules_settings = {}
        self.d5a_modules_settings = {}
        self.dacs = {}
        self.device_parameters = {}

    def connect(self):
        """Connects to the instrument using the IP address set in the
        runcard."""
        if not self.is_connected:
            for attempt in range(3):
                try:
                    self.device = SpiRack(self.name, self.address)
                    self.is_connected = True
                    break
                except KeyError as exc:
                    log.info(f"Unable to connect:\n{str(exc)}\nRetrying...")
                    self.name += "_" + str(attempt)
                except Exception as exc:
                    log.info(f"Unable to connect:\n{str(exc)}\nRetrying...")
            if not self.is_connected:
                raise InstrumentException(self, f"Unable to connect to {self.name}")
        else:
            raise_error(
                Exception, "There is an open connection to the instrument already"
            )

    def _set_device_parameter(self, target, *parameters, value):
        if self.is_connected:
            key = target.name + "." + parameters[0]
            if not key in self.device_parameters:
                for parameter in parameters:
                    if not hasattr(target, parameter):
                        raise Exception(
                            f"The instrument {self.name} does not have parameters {parameter}"
                        )
                    target.set(parameter, value)
                self.device_parameters[key] = value
            elif self.device_parameters[key] != value:
                for parameter in parameters:
                    target.set(parameter, value)
                self.device_parameters[key] = value
        else:
            raise Exception("There is no connection to the instrument {self.name}")

    def setup(self, **kwargs):
        # Init S4g and D5a modules in SPI mapped on runcard
        if self.is_connected:
            # TODO: Check data format from yml
            #      Make d5g modules optional in runcard
            #      Define span values in setup
            #      Implement parameters cache
            #      export current / voltage properties (and make them sweepable)
            if "s4g_modules" in kwargs:
                self.s4g_modules_settings = kwargs["s4g_modules"]
            if "d5a_modules" in kwargs:
                self.d5a_modules_settings = kwargs["d5a_modules"]

            for channel, settings in self.s4g_modules_settings.items():
                module_number = settings[0]
                port_number = settings[1]
                module_name = f"S4g_module{module_number}"
                current = settings[2]
                if not module_name in self.device.instrument_modules:
                    self.device.add_spi_module(settings[0], "S4g", module_name)
                device = self.device.instrument_modules[module_name].instrument_modules[
                    "dac" + str(port_number - 1)
                ]
                self.dacs[channel] = type(
                    "S4g_dac",
                    (),
                    {
                        "current": self.property_wrapper(device, "current"),
                        "device": device,
                    },
                )()
                self.dacs[channel].device.span("range_min_bi")
                # self.dacs[channel].current = current

            for channel, settings in self.d5a_modules_settings.items():
                module_number = settings[0]
                port_number = settings[1]
                module_name = f"D5a_module{module_number}"
                voltage = settings[2]
                if not module_name in self.device.instrument_modules:
                    self.device.add_spi_module(settings[0], "D5a", module_name)
                device = self.device.instrument_modules[module_name].instrument_modules[
                    "dac" + str(port_number - 1)
                ]
                self.dacs[channel] = type(
                    "D5a_dac",
                    (),
                    {
                        "voltage": self.property_wrapper(device, "voltage"),
                        "device": device,
                    },
                )()
                self.dacs[channel].device.span("range_min_bi")
                # self.dacs[channel].voltage = voltage
        else:
            raise_error(Exception, "There is no connection to the instrument")

    def set_SPI_DACS_to_cero(self):
        self.device.set_dacs_zero()

    def get_SPI_IDN(self):
        return self.device.IDN()

    def get_SPI_temperature(self):
        return self.device.temperature()

    def get_SPI_battery_voltage(self):
        return self.device.battery_voltages()

    def disconnect(self):
        if self.is_connected:
            self.is_connected = False

    def close(self):
        if self.is_connected:
            self.device.close()
            self.is_connected = False

    def start(self):
        # Set the dacs to the values stored for each qubit in the runcard
        if self.is_connected:
            for channel, settings in self.s4g_modules_settings.items():
                current = settings[2]
                # Check current current of the module and warning
                if abs(self.dacs[channel].current) > 0.010:
                    log.info(
                        f"WARNING: S4g module {settings[0]} - port {settings[1]} current was: {self.dacs[channel].current}, now setting current to: {current}"
                    )
                self.dacs[channel].current = current

            for channel, settings in self.d5a_modules_settings.items():
                voltage = settings[2]
                # Check current current of the module and warning
                if abs(self.dacs[channel].voltage) > 0.010:
                    log.info(
                        f"WARNING: D5a module {settings[0]} - port {settings[1]} voltage was: {self.dacs[channel].voltage}, now setting voltage to: {voltage}"
                    )
                self.dacs[channel].voltage = voltage

    def stop(self):
        # if self.is_connected:
        #     self.device.set_dacs_zero()
        return
