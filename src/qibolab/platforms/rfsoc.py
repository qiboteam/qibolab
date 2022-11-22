import yaml
from qibo.config import log
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform


class RFSocPlatform(AbstractPlatform):
    def __init__(self, name, runcard):
        log.info(f"Loading platform {name} from runcard {runcard}")
        self.name = name
        self.runcard = runcard
        self.is_connected = False
        # Load platform settings
        with open(runcard) as file:
            self.settings = yaml.safe_load(file)

        self.nqubits = self.settings.get("nqubits")
        if self.nqubits == 1:
            self.resonator_type = "3D"
        else:
            self.resonator_type = "2D"

#        self.cfg = dict(self.settings.get("config"))
        self.cfg = dict(self.settings.get("instruments"))
        self.host = self.cfg["tii_rfsoc4x2"]["settings"]["ip_address"]
        self.port = self.cfg["tii_rfsoc4x2"]["settings"]["ip_port"]
        print("Configuracion: ", self.cfg)
    def reload_settings(self):
        raise NotImplementedError

    def run_calibration(self, show_plots=False):  # pragma: no cover
        raise NotImplementedError

    def connect(self):
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def disconnect(self):
        raise NotImplementedError

    def _generate_json_packet(self, var1):
        # Generate the json packet to send over TCP
        return {
            "opcode": 3, 
            "length": var1
            }
        

    def execute_pulse_sequence(self, sequence, rabi_length,  nshots=None):
        import socket
        import json

        self.cfg["tii_rfsoc4x2"]["settings"]["rabi_length"] = rabi_length

        jsonDic = self.cfg 

    # Create a socket (SOCK_STREAM means a TCP socket)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to server and send data
            sock.connect((self.host, self.port))
            sock.sendall(json.dumps(jsonDic).encode())
            print("datos enviados")
            # Receive data from the server and shut down
            received = sock.recv(1024)
            avg = json.loads(received.decode('utf-8'))          
            avgi = np.array([avg["avgiRe"], avg["avgiIm"]])
            avgq = np.array([avg["avgqRe"], avg["avgqIm"]])
        sock.close()

        return avgi, avgq
    

