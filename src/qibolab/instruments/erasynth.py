import json

import requests
from qcodes_contrib_drivers.drivers.ERAInstruments import ERASynthPlusPlus
from qibo.config import log

from qibolab.instruments.oscillator import LocalOscillator

RECONNECTION_ATTEMPTS = 10
"""Number of times to attempt sending requests to the web server in case of
failure."""
TIMEOUT = 10
"""Timeout time for HTTP requests in seconds."""


class ERASynthEthernet:
    """ERA ethernet driver that follows the QCoDeS interface.

    Controls the instrument via HTTP requests to the instrument's web
    server.
    """

    def __init__(self, name, address):
        self.name = name
        self.address = address
        self.post("readAll", 1)
        self.post("readDiagnostic", 0)
        self.post("rfoutput", 0)

    @property
    def url(self):
        return f"http://{self.address}/"

    def post(self, name, value):
        """Post a value to the instrument's web server.

        Try to post multiple times, waiting for 0.1 seconds between each attempt.

        Args:
            name: str = The name of the value to post.
            value: str = The value to post.
        """
        value = str(value)
        for _ in range(RECONNECTION_ATTEMPTS):
            try:
                response = requests.post(self.url, data={name: value}, timeout=TIMEOUT)
                if response.status_code == 200:
                    return True
                break
            except (ConnectionError, TimeoutError, requests.exceptions.ReadTimeout):
                log.info("ERAsynth connection timed out, retrying...")
        raise ConnectionError(f"Unable to post {name}={value} to {self.name}")

    def get(self, name):
        """Get a value from the instrument's web server.

        Try to get multiple times, waiting for 0.1 seconds between each attempt.

        Args:
            name: str = The name of the value to get.
        """
        if name == "ref_osc_source":
            value = self.get("reference_int_ext")
            if value == 1:
                return "EXT"
            else:
                return "INT"

        for _ in range(RECONNECTION_ATTEMPTS):
            try:
                response = requests.post(
                    self.url, params={"readAll": 1}, timeout=TIMEOUT
                )
                if response.status_code == 200:
                    # reponse.text is a dictonary in string format, convert it to a dictonary
                    return json.loads(response.text)[name]
                break
            except (ConnectionError, TimeoutError, requests.exceptions.ReadTimeout):
                log.info("ERAsynth connection timed out, retrying...")

        raise ConnectionError(f"Unable to get {name} from {self.name}")

    def set(self, name, value):
        """Set a value to the instrument's web server.

        Args:
            name (str): Name of the paramater that we are updating.
                In qibolab this can be ``frequency``, ``power`` or ``ref_osc_source``,
                however the instrument's web server may support more values.
            value: New value to set to the given parameter.
                The type of value depends on the parameter being updated.
        """
        if name == "ref_osc_source":
            if value.lower() in ("int", "internal"):
                self.post("reference_int_ext", 0)
            elif value.lower() in ("ext", "external"):
                self.post("reference_int_ext", 1)
            else:
                raise ValueError(f"Invalid reference clock source {value}")

        elif name == "frequency":
            self.post(name, int(value))

        elif name == "power":
            self.post(name, float(value))

        else:
            self.post(name, value)

    def on(self):
        self.post("rfoutput", 1)

    def off(self):
        self.post("rfoutput", 0)

    def close(self):
        self.off()


class ERA(LocalOscillator):
    """Driver to control the ERAsynth++ local oscillator.

    This driver is using:
    https://qcodes.github.io/Qcodes_contrib_drivers/api/generated/qcodes_contrib_drivers.drivers.ERAInstruments.html#qcodes_contrib_drivers.drivers.ERAInstruments.erasynth.ERASynthPlusPlus

    or the custom :class:`qibolab.instruments.erasynth.ERASynthEthernet` object
    if we are connected via ethernet.
    """

    def __init__(self, name, address, ethernet=True, ref_osc_source=None):
        super().__init__(name, address, ref_osc_source)
        self.ethernet = ethernet

    def create(self):
        if self.ethernet:
            return ERASynthEthernet(self.name, self.address)
        else:
            return ERASynthPlusPlus(f"{self.name}", f"TCPIP::{self.address}::INSTR")

    def __del__(self):
        self.disconnect()
