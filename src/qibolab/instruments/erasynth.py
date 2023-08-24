"""ERAsynth drivers.

Supports the ERAsynth ++.

https://qcodes.github.io/Qcodes_contrib_drivers/_modules/qcodes_contrib_drivers/drivers/ERAInstruments/erasynth.html#ERASynthBase.clear_read_buffer
"""

import json

import requests
from qcodes_contrib_drivers.drivers.ERAInstruments import ERASynthPlusPlus
from qibo.config import log

from qibolab.instruments.abstract import InstrumentException
from qibolab.instruments.oscillator import LocalOscillator

MAX_RECONNECTION_ATTEMPTS = 10
TIMEOUT = 10


class ERA(LocalOscillator):
    def __init__(self, name, address, ethernet=True, reference_clock_source="internal"):
        super().__init__(name, address)
        self.device: ERASynthPlusPlus = None
        self._power: int = None
        self._frequency: int = None
        self.ethernet = ethernet
        self._device_parameters = {}
        self.reference_clock_source = reference_clock_source

    @property
    def frequency(self):
        if self.is_connected:
            if self.ethernet:
                return int(self._get("frequency"))
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
            if self.ethernet:
                return float(self._get("amplitude"))
            return self.device.get("power")
        return self._power

    @power.setter
    def power(self, x):
        self._power = x
        if self.is_connected:
            self._set_device_parameter("power", x)

    def connect(self):
        """Connects to the instrument using the IP address set in the runcard."""
        if not self.is_connected:
            for attempt in range(3):
                try:
                    if not self.ethernet:
                        self.device = ERASynthPlusPlus(f"{self.name}", f"TCPIP::{self.address}::INSTR")
                    else:
                        self._post("readAll", 1)
                        self._post("readDiagnostic", 0)
                    self.is_connected = True
                    break
                except KeyError as exc:
                    print(f"Unable to connect:\n{str(exc)}\nRetrying...")
                    self.name += "_" + str(attempt)
                except ConnectionError as exc:
                    print(f"Unable to connect:\n{str(exc)}\nRetrying...")
            if not self.is_connected:
                raise InstrumentException(self, f"Unable to connect to {self.name}")
        else:
            raise RuntimeError("There is an open connection to the instrument already")
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
                if not self.ethernet:
                    if not hasattr(self.device, parameter):
                        raise ValueError(f"The instrument {self.name} does not have parameter {parameter}")
                    self.device.set(parameter, value)
                else:
                    if parameter == "power":
                        self._post("amplitude", float(value))
                    elif parameter == "frequency":
                        self._post("frequency", int(value))
                self._device_parameters[parameter] = value
            else:
                raise ConnectionError(f"Attempting to set {parameter} without a connection to the instrument")

    def _erase_device_parameters_cache(self):
        """Erases the cache of the instrument parameters."""
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

            if not "reference_clock_source" in kwargs:
                kwargs["reference_clock_source"] = self.reference_clock_source
            if not self.ethernet:
                if kwargs["reference_clock_source"] == "internal":
                    self.device.ref_osc_source("int")
                elif kwargs["reference_clock_source"] == "external":
                    self.device.ref_osc_source("ext")
                else:
                    raise ValueError(f"Invalid reference clock source {kwargs['reference_clock_source']}")
            else:
                self._post("rfoutput", 0)

                if kwargs["reference_clock_source"] == "internal":
                    self._post("reference_int_ext", 0)
                elif kwargs["reference_clock_source"] == "external":
                    self._post("reference_int_ext", 1)
                else:
                    raise ValueError(f"Invalid reference clock source {kwargs['reference_clock_source']}")
        else:
            raise ConnectionError("There is no connection to the instrument")

    def start(self):
        self.on()

    def stop(self):
        self.off()

    def disconnect(self):
        if self.is_connected:
            self.is_connected = False

    def __del__(self):
        self.disconnect()

    def on(self):
        if not self.ethernet:
            self.device.on()
        else:
            self._post("rfoutput", 1)

    def off(self):
        if not self.ethernet:
            self.device.off()
        else:
            self._post("rfoutput", 0)

    def close(self):
        self.is_connected = False

    def _post(self, name, value):
        """
        Post a value to the instrument's web server.

        Try to post three times, waiting for 0.1 seconds between each attempt.

        Args:
            name: str = The name of the value to post.
            value: str = The value to post.
        """
        value = str(value)
        for _ in range(MAX_RECONNECTION_ATTEMPTS):
            try:
                response = requests.post(f"http://{self.address}/", data={name: value}, timeout=TIMEOUT)
                if response.status_code == 200:
                    return True
                break
            except (ConnectionError, TimeoutError, requests.exceptions.ReadTimeout):
                log.info("ERAsynth connection timed out, retrying...")
        raise InstrumentException(self, f"Unable to post {name}={value} to {self.name}")

    def _get(self, name):
        """
        Get a value from the instrument's web server.

        Try to get three times, waiting for 0.1 seconds between each attempt.

        Args:
            name: str = The name of the value to get.
        """
        for _ in range(MAX_RECONNECTION_ATTEMPTS):
            try:
                response = requests.post(f"http://{self.address}/", params={"readAll": 1}, timeout=TIMEOUT)

                if response.status_code == 200:
                    # reponse.text is a dictonary in string format, convert it to a dictonary
                    return json.loads(response.text)[name]
                break
            except (ConnectionError, TimeoutError, requests.exceptions.ReadTimeout):
                log.info("ERAsynth connection timed out, retrying...")
        raise InstrumentException(self, f"Unable to get {name} from {self.name}")
