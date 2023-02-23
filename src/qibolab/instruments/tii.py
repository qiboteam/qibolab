""" RFSoC fpga driver.
Supports the following FPGA:
    RFSoC 4x2
"""
import json
import socket
import time

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
    PulseType,
    ReadoutPulse,
    Rectangular,
)
from qibolab.sweeper import Parameter, Sweeper


class TII_RFSOC4x2(AbstractInstrument):
    def __init__(self, name: str, address: str):  # , setting_parameters: dict):
        super().__init__(name, address)
        self.cfg: dict = {}
        self.host: str
        self.port: str
        self.host, port = address.split(":")
        self.port = int(port)

    def connect(self):
        self.is_connected = True

        # Create a socket (SOCK_STREAM means a TCP socket) and send configuration
        jsonDic = self.cfg
        jsonDic["opCode"] = "configuration"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to server and send data
            sock.connect((self.host, self.port))
            sock.sendall(json.dumps(jsonDic).encode())
        sock.close()

    def setup(self, qubits, **kwargs):
        """Configures the instrument.
        A connection to the instrument needs to be established beforehand.
        Args:
            **kwargs: dict = A dictionary of settings loaded from the runcard:
                kwargs['hardware_avg']
                kwargs['repetition_duration']
                kwargs['sampling_rate']
                kwargs['resonator_phase']
                kwargs['adc_trig_offset']
                kwargs['threshold']
                kwargs['relax_delay']
                kwargs['max_gain']
        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """

        if self.is_connected:
            # Load settings
            self.cfg = kwargs
            jsonDic = self.cfg
            # print("Check point 3", jsonDic)
            jsonDic["opCode"] = "setup"
            # Create a socket (SOCK_STREAM means a TCP socket)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                # Connect to server and send data
                sock.connect((self.host, self.port))
                sock.sendall(json.dumps(jsonDic).encode())
            sock.close()

        else:
            raise Exception("The instrument cannot be set up, there is no connection")

    def play(self, qubits, sequence, nshots, relaxation_time):
        """Executes the sequence of instructions and retrieves the readout results.
        Each readout pulse generates a separate acquisition.
        Returns:
            Two array with real and imaginary parts if i and q (already averages)
        TODO: Refactor this method according to sweep method
        """
        ps: PulseShape
        jsonDic = {}
        jsonDic["opCode"] = "execute"
        pulsesDic = {}
        for i, pulse in enumerate(sequence):
            pulsesDic[str(i)] = self.convert_pulse_to_dic(pulse)
        jsonDic["pulses"] = pulsesDic

        # Create a socket (SOCK_STREAM means a TCP socket)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to server and send data
            sock.connect((self.host, self.port))
            sock.sendall(json.dumps(jsonDic).encode())
            # Receive data from the server and shut down
            received = sock.recv(256)
            avg = json.loads(received.decode("utf-8"))
            avgi = avg["avgi"]
            avgq = avg["avgq"]
        sock.close()
        return avgi, avgq

    def sweep(self, qubits, sequence, *sweepers, nshots, relaxation_time, average=True):
        """Play a pulse sequence while sweeping one or more parameters."""

        #  Parsing the sweeper to dictionary and after to a json file
        s: Sweeper
        par: Parameter
        s = sweepers[0]

        jsonDic = {}
        jsonDic["parameter"] = str(s.parameter)
        start = s.values[0].item()
        expt = len(s.values)

        step = (s.values[1] - s.values[0]).item()
        range = {"start": start, "step": step, "expt": expt}

        jsonDic["range"] = range

        pulsesDic = {}
        for i, pulse in enumerate(s.pulses):
            pulsesDic[str(i)] = self.convert_pulse_to_dic(pulse)
        jsonDic["pulses"] = pulsesDic

        jsonDic["opCode"] = "sweep"

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        received = bytearray()
        try:
            # connect to server
            sock.connect((self.host, self.port))

            # send data
            sock.sendall(json.dumps(jsonDic).encode())

            # receive data back from the server
            while 1:
                tmp = sock.recv(4096)
                if not tmp:
                    break
                received.extend(tmp)
            # received = sock.recv(65536)
            # time.sleep(1)
            avg = json.loads(received.decode("utf-8"))
            avg_di = avg["avg_di"]
            avg_dq = avg["avg_dq"]
        finally:
            # shut down
            sock.close()

        # Create a socket (SOCK_STREAM means a TCP socket)
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to server and send data
            sock.connect((self.host, self.port))
            sock.sendall(json.dumps(jsonDic).encode())
            # Receive data from the server and shut down
            received = sock.recv(65536)
            avg = json.loads(received.decode("utf-8"))
            avg_di = avg["avg_di"]
            avg_dq = avg["avg_dq"]
        sock.close()
        """
        return avg_di, avg_dq

    def convert_pulse_to_dic(self, pulse):
        """Funtion to convert pulse object attributes to a dictionary"""
        p: Pulse
        ps: PulseShape
        pulseDic = {}
        pDic = {}
        p = pulse

        if pulse.type == PulseType.DRIVE:
            ps = pulse.shape
            if type(ps) is Drag:
                shape = "Drag"
                style = "arb"
                rel_sigma = ps.rel_sigma
                beta = ps.beta
            elif type(ps) is Gaussian:
                shape = "Gaussian"
                style = "arb"
                rel_sigma = ps.rel_sigma
                beta = 0
            elif type(ps) is Rectangular:
                shape = "Rectangular"
                style = "const"
                rel_sigma = 0
                beta = 0
            pDic = {
                "start": pulse.start,
                "duration": pulse.duration,
                "amplitude": pulse.amplitude,
                "frequency": pulse.frequency,
                "relative_phase": pulse.relative_phase,
                "shape": shape,
                "style": style,
                "rel_sigma": rel_sigma,
                "beta": beta,
                "type": "qd",
            }

        elif pulse.type == PulseType.READOUT:
            pDic = {
                "start": pulse.start,
                "duration": pulse.duration,
                "amplitude": pulse.amplitude,
                "frequency": pulse.frequency,
                "relative_phase": pulse.relative_phase,
                "shape": "const",
                "type": "ro",
            }

        return pDic

    def start(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def stop(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def disconnect(self):
        """Closes the connection to the instrument."""
        self.is_connected = False
