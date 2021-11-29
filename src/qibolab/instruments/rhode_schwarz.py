"""
Class to interface with the local oscillator RohdeSchwarz SGS100A
"""

import logging
import qcodes.instrument_drivers.rohde_schwarz.SGS100A as LO_SGS100A

logger = logging.getLogger(__name__)  # TODO: Consider using a global logger


class SGS100A(LO_SGS100A.RohdeSchwarz_SGS100A):

    def __init__(self, label, ip):
        """
        create Local Oscillator with name = label and connect to it in local IP = ip
        Params format example:
                "ip": '192.168.0.8',
                "label": "qcm_LO"
        """
        super().__init__(label, f"TCPIP0::{ip}::inst0::INSTR")
        logger.info("Local oscillator connected")

    def setup(self, power, frequency):
        #set dbm power to Local Oscillator
        self.power(power)
        logger.info(f"Local oscillator power set to {power}.")
        self.frequency(frequency)
        logger.info(f"Local oscillator frequency set to {frequency}.")

    def on(self):
        """Start generating microwaves."""
        self.on()
        logger.info("Local oscillator on.")

    def off(self):
        """Stop generating microwaves."""
        self.off()
        logger.info("Local oscillator off.")
