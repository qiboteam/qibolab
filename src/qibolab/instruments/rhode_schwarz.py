"""
Class to interface with the local oscillator RohdeSchwarz SGS100A
"""

import logging
import qcodes.instrument_drivers.rohde_schwarz.SGS100A as LO_SGS100A

logger = logging.getLogger(__name__)  # TODO: Consider using a global logger


class SGS100A:

    def __init__(self, label, ip):
        """
        create Local Oscillator with name = label and connect to it in local IP = ip
        Params format example:
                "ip": '192.168.0.8',
                "label": "qcm_LO"
        """
        self.device = LO_SGS100A.RohdeSchwarz_SGS100A(label, f"TCPIP0::{ip}::inst0::INSTR")
        logger.info("Local oscillator connected")
        self._power = None
        self._frequency = None

    def setup(self, power, frequency):
        self.set_power(power)
        self.set_frequency(frequency)

    def set_power(self, power):
        """Set dbm power to local oscillator."""
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
        self.device.close()

    def __del__(self):
        self.device.close()
