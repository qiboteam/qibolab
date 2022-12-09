""" RFSoC fpga driver.

Supports the following FPGA:
    RFSoC 4x2
"""
import json
import socket

import numpy as np
import yaml
from qibo.config import log

from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.pulses import (
    Drag,
    Gaussian,
    Pulse,
    PulseSequence,
    PulseShape,
    ReadoutPulse,
    Rectangular,
)


#################################################################
#
class tii_rfsoc4x2:
    #
    #################################################################
    def __init__(self, name: str, address: str, setting_parameters: dict):

        self.settings = setting_parameters
        self.is_connected = False
        self.cfg = self.settings["instruments"]["tii_rfsoc4x2"]["settings"]
        self.host = self.cfg["ip_address"]
        self.port = self.cfg["ip_port"]

        # Create a socket (SOCK_STREAM means a TCP socket) and send configuration
        jsonDic = self.cfg
        jsonDic["opCode"] = "configuration"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to server and send data
            sock.connect((self.host, self.port))
            sock.sendall(json.dumps(jsonDic).encode())
        sock.close()

    def setup(self):
        self.experiment = self.settings["instruments"]["tii_rfsoc4x2"]["experiment"]

        jsonDic = self.experiment
        jsonDic["opCode"] = "setup"
        # Create a socket (SOCK_STREAM means a TCP socket)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to server and send data
            sock.connect((self.host, self.port))
            sock.sendall(json.dumps(jsonDic).encode())

        sock.close()

    def play_sequence_and_acquire(self, sequence):
        """Executes the sequence of instructions and retrieves the readout results.

        Each readout pulse generates a separate acquisition.
        Returns:
            Two array with real and imaginary parts if i and q (already averages)

        """
        ps: PulseShape

        jsonDic = {}
        i = 0
        for pulse in sequence:
            ps = pulse.shape
            if type(ps) is Drag:
                shape = "Drag"
                style = "arb"
                rel_sigma = ps.rel_sigma
                beta = ps.beta
            elif type(ps) is Gaussian:
                shape: "Gaussian"
                style = "arb"
                rel_sigma = ps.rel_sigma
                beta = 0
            elif type(ps) is Rectangular:
                shape: "Rectangular"
                style = "const"
                rel_sigma = 0
                beta = 0

            pulseDic = {
                "start": pulse.start,
                "duration": pulse.duration,
                "amplitude": pulse.amplitude,
                "frequency": pulse.frequency,
                "relative_phase": pulse.relative_phase,
                #                        "shape": shape,
                "style": style,
                "rel_sigma": rel_sigma,
                "beta": beta,
                "channel": pulse.channel,
                #                        "type": pulse.type,
                "qubit": pulse.qubit,
            }
            jsonDic["pulse" + str(i)] = pulseDic
            i = i + 1

        jsonDic["opCode"] = "execute"
        # Create a socket (SOCK_STREAM means a TCP socket)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to server and send data
            sock.connect((self.host, self.port))
            sock.sendall(json.dumps(jsonDic).encode())
            # Receive data from the server and shut down
            received = sock.recv(256)
            avg = json.loads(received.decode("utf-8"))
            avgi = np.array([avg["avgiRe"], avg["avgiIm"]])
            avgq = np.array([avg["avgqRe"], avg["avgqIm"]])
        sock.close()
        return avgi, avgq
