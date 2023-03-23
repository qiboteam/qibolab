"""RohdeSchwarz driver.

Supports the following Instruments:
    SGS100A

https://qcodes.github.io/Qcodes/api/generated/qcodes.instrument_drivers.rohde_schwarz.html#module-qcodes.instrument_drivers.rohde_schwarz.SGS100A
"""
import qcodes.instrument_drivers.rohde_schwarz.SGS100A as LO_SGS100A

from qibolab.instruments.abstract import InstrumentException, LocalOscillator


class SGS100A(LocalOscillator):
    def __init__(self, name, address):
        super().__init__(name, address)
        self.device: LO_SGS100A = None
        self._power: float = None
        self._frequency: float = None
        self._device_parameters = {}

    @property
    def frequency(self):
        if self.is_connected:
            return self.device.get("frequency")
        return self._frequency

    @frequency.setter
    def frequency(self, x):
        self._frequency = x
        if self.is_connected:
            self._set_device_parameter("frequency", x)

    @property
    def power(self):
        if self.is_connected:
            return self.device.get("power")
        return self._power

    @power.setter
    def power(self, x):
        self._power = x
        if self.is_connected:
            self._set_device_parameter("power", x)

    def connect(self):
        """
        Connects to the instrument using the IP address set in the runcard.
        """
        if not self.is_connected:
            for attempt in range(3):
                try:
                    self.device = LO_SGS100A.RohdeSchwarz_SGS100A(self.name, f"TCPIP0::{self.address}::5025::SOCKET")
                    self.is_connected = True
                    break
                except KeyError as exc:
                    print(f"Unable to connect:\n{str(exc)}\nRetrying...")
                    self.name += "_" + str(attempt)
                except Exception as exc:
                    print(f"Unable to connect:\n{str(exc)}\nRetrying...")
            if not self.is_connected:
                raise InstrumentException(self, f"Unable to connect to {self.name}")
        else:
            raise Exception("There is an open connection to the instrument already")
        # set proper frequency and power if they were changed before connecting
        if self._frequency is not None:
            self._set_device_parameter("frequency", self._frequency)
        if self._power is not None:
            self._set_device_parameter("power", self._power)

    def _set_device_parameter(self, parameter: str, value):
        """Sets a parameter of the instrument, if it changed from the last stored in the cache.

        Args:
            parameter: str = The parameter to be cached and set.
            value = The value to set the paramter.
        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """
        if not (parameter in self._device_parameters and self._device_parameters[parameter] == value):
            if self.is_connected:
                if not parameter in self._device_parameters:
                    if not hasattr(self.device, parameter):
                        raise Exception(f"The instrument {self.name} does not have parameter {parameter}")
                    self.device.set(parameter, value)
                    self._device_parameters[parameter] = value
                elif self._device_parameters[parameter] != value:
                    self.device.set(parameter, value)
                    self._device_parameters[parameter] = value
            else:
                raise Exception("There is no connection to the instrument {self.name}")

    def _erase_device_parameters_cache(self):
        """Erases the cache of instrument parameters."""
        self._device_parameters = {}

    def setup(self, frequency=None, power=None, **kwargs):
        """Configures the instrument.

        A connection to the instrument needs to be established beforehand.
        Args:
            **kwargs: dict = A dictionary of settings loaded from the runcard:
                kwargs["power"]
                kwargs["frequency"]
        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """
        if frequency is None:
            frequency = self.frequency
        if power is None:
            power = self.power

        if self.is_connected:
            # Load settings
            self.power = power
            self.frequency = frequency
        else:
            raise Exception("There is no connection to the instrument")

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
