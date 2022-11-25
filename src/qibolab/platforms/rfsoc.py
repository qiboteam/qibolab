import yaml
from qibo.config import log
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import (
    Drag,
    Gaussian,
    Pulse,
    PulseSequence,
    ReadoutPulse,
    Rectangular,
)
import socket
import json

class RFSocPlatform(AbstractPlatform):
    def __init__(self, name, runcard):
        log.info(f"Loading platform {name} from runcard {runcard}")
        self.name = name
        self.runcard = runcard
        self.is_connected = False
        # Load platform settings
        with open(runcard) as file:
            self.settings = yaml.safe_load(file)


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

    def reload_settings(self):
        raise NotImplementedError

    def run_calibration(self, show_plots=False):  # pragma: no cover
        raise NotImplementedError

    def connect(self):      
        raise NotImplementedError

    def setup(self, rabi_length):
        self.experiment = self.settings["instruments"]["tii_rfsoc4x2"]["experiment"]
        self.experiment["rabi_length"] = rabi_length

        jsonDic = self.experiment 
        jsonDic["opCode"] = "setup"
        # Create a socket (SOCK_STREAM means a TCP socket)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to server and send data
            sock.connect((self.host, self.port))
            sock.sendall(json.dumps(jsonDic).encode())

        sock.close()
        return  

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def disconnect(self):
        raise NotImplementedError



    def execute_pulse_sequence(self, sequence,   nshots=None):
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

            pulseDic = {"start": pulse.start, 
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
                        "qubit": pulse.qubit
                        }
            jsonDic['pulse'+str(i)] = pulseDic
            i = i+1

        jsonDic["opCode"] = "execute"
        # Create a socket (SOCK_STREAM means a TCP socket)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to server and send data
            sock.connect((self.host, self.port))
            sock.sendall(json.dumps(jsonDic).encode())
            # Receive data from the server and shut down
            received = sock.recv(256)
            avg = json.loads(received.decode('utf-8'))          
            avgi = np.array([avg["avgiRe"], avg["avgiIm"]])
            avgq = np.array([avg["avgqRe"], avg["avgqIm"]])
        sock.close()

        return avgi, avgq
  

