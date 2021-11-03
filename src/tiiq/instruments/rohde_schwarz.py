"""
CLASS TO INTERFACE WITH THE LOCAL OSCILLATOR RohdeSchwarz SGS100A
"""

import logging
import qcodes.instrument_drivers.rohde_schwarz.SGS100A as LO_SGS100A

logger = logging.getLogger(__name__)

class SGS100A():

    def __init__(self, label, ip):
        """
        create Local Oscillator with name = label and connect to it in local IP = ip
        Params format example:
                "ip": 'TCPIP0::192.168.0.8::inst0::INSTR',
                "label": "qcm_LO"
        """
        self.LO = LO_SGS100A.RohdeSchwarz_SGS100A(label, 'TCPIP0::'+ip+'::inst0::INSTR')
        self.ip = ip
        self.label = label
        logger.info("Local oscillator connected")

    def set_power(self, power):
        #set dbm power to Local Oscillator
        self.LO.power(power)
        self.power = power
        logger.info("Local oscillator power set")

    def set_frequency(self, frequency):
        #set dbm frequency to Local oscillator
        self.LO.frequency(frequency)
        self.frequency = frequency
        logger.info("Local oscillator frequency set")

    def on(self):
        # start generating microwaves
        self.LO.on()
        self.state = "ON"
        logger.info("Local oscillator on")

    def off(self):
        # stop generating microwaves
        self.LO.off()
        self.state = "OFF"
        logger.info("Local oscillator off")

    def getName(self):
        # return actual LO name
        return self.label

    def getIP(self):
        # return actual LO network ip
        return self.ip


    def getPower(self):
        # return actual LO power
        return self.power

    def getFrequency(self):
        # return actual LO frquency
        return self.frequency
