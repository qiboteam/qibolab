"""
Class to interface with the local oscillator RohdeSchwarz SGS100A
"""

import logging
from .instrument import Instrument
import qcodes.instrument_drivers.rohde_schwarz.SGS100A as LO_SGS100A

logger = logging.getLogger(__name__)  # TODO: Consider using a global logger


class SGS100A(Instrument):

    def __init__(self, label, ip):
        """
        create Local Oscillator with name = label and connect to it in local IP = ip
        Params format example:
                "ip": '192.168.0.8',
                "label": "qcm_LO"
        """
        self._power  = None
        self._frequency = None
        self._connected = False
        self._signature = f"{type(self).__name__}@{ip}"
        self.label = label
        self.ip = ip
        self.connect()

    def connect(self):
        try:
            self._driver = LO_SGS100A.RohdeSchwarz_SGS100A(self.label, f"TCPIP0::{self.ip}::inst0::INSTR")
        except Exception as exc:
            print(f"Can't connect to SGS100A at ip {ip}.")
            print(exc) 
            raise exc
        self._connected = True
        logger.info("Local oscillator connected")

    def setup(self, power, frequency):
        self.set_power(power)
        self.set_frequency(frequency)

    def set_power(self, power):
        """Set dBm power to local oscillator."""
        self._power = power
        self.device.power(power)
        logger.info(f"Local oscillator power set to {power}.")

    def set_frequency(self, frequency):
        self._frequency = frequency
        self.device.frequency(frequency)
        logger.info(f"Local oscillator frequency set to {frequency}.")

    def get_power(self):
        if self._power is not None:
            return self._power
        raise RuntimeError("Local oscillator power was not set.")

    def get_frequency(self):
        if self._frequency is not None:
            return self._frequency
        raise RuntimeError("Local oscillator frequency was not set.")

    def on(self):
        """Start generating microwaves."""
        self.device.on()
        logger.info("Local oscillator on.")

    def off(self):
        """Stop generating microwaves."""
        self.device.off()
        logger.info("Local oscillator off.")

    def close(self):
        if self._connected:
            self.off()
            self.device.close()
            self._connected = False

    # TODO: Figure out how to fix this
    #def __del__(self):
    #    self.close()
