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
    """Instrument object for controlling the RFSoC4x2 FPGA.

    The connection requires the FPGA to have a server currently listening.
    The ``setup`` function must be called before playing pulses with
    ``play`` (for arbitrary qibolab ``PulseSequence``) or ``sweep``.

    Args:
        name (str): Name of the instrument instance.
        address (str): IP address and port for connecting to the FPGA.
    """
    def __init__(self, name: str, address: str):
        super().__init__(name, address)
        self.cfg: dict = {}
        self.host: str
        self.port: str
        self.host, port = address.split(":")
        self.port = int(port)

    def connect(self):
        """Connects to the FPGA instrument."""
        self.is_connected = True

        # Create a socket (SOCK_STREAM means a TCP socket) and send configuration
        jsonDic = self.cfg
        jsonDic["opCode"] = "configuration" # opCode parameter for server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            sock.sendall(json.dumps(jsonDic).encode())

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
            jsonDic["opCode"] = "setup" # opCode parameter for server
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.host, self.port))
                sock.sendall(json.dumps(jsonDic).encode())

        else:
            raise Exception("The instrument cannot be set up, there is no connection")

    def play(self, qubits, sequence, nshots, relaxation_time):
        """Executes the sequence of instructions and retrieves the readout results.

        Each readout pulse generates a separate acquisition.

        Args:
            nshots (int): parameter not used
            relaxation_time: parameter not used

        Returns:
            Two array with real and imaginary parts if i and q (already averages)
        """
        # TODO clean not used parameters

        jsonDic = {}
        jsonDic["opCode"] = "execute"
        pulsesDic = {}
        for i, pulse in enumerate(sequence):
            pulsesDic[str(i)] = self.convert_pulse_to_dic(pulse)
        jsonDic["pulses"] = pulsesDic

        # Create a socket and send pulse sequence to the FPGA
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            sock.sendall(json.dumps(jsonDic).encode())
            # read from the server a maximum of 256 bytes (enough for sequence)
            received = sock.recv(256)
            #avg = json.loads(received)
            avg = json.loads(received.decode("utf-8"))
            avgi = avg["avgi"]
            avgq = avg["avgq"]
        return avgi, avgq

    def sweep(self, qubits, sequence, *sweepers, nshots, relaxation_time, average=True):
        """Play a pulse sequence while sweeping one or more parameters.

        Args:
            qubits: parameter not used
            sequence: parameter not used
            *sweepers (list): A list of qibolab Sweepers objects
            nshots (int): parameter not used
            relaxation_time: parameter not used
            average: parameter not used
        """
        # TODO clean not used parameters

        #  Parsing the sweeper to dictionary and after to a json file
        s: Sweeper
        par: Parameter
        s = sweepers[0]

        jsonDic = {}
        jsonDic["parameter"] = str(s.parameter)
        start = s.values[0].item()
        expt = len(s.values)

        step = (s.values[1] - s.values[0]).item()
        jsonDic["range"] = {"start": start, "step": step, "expt": expt}

        pulsesDic = {}
        for i, pulse in enumerate(sequence.pulses): # convert pulses to dictionary
            pulsesDic[str(i)] = self.convert_pulse_to_dic(pulse)
        jsonDic["pulses"] = pulsesDic

        jsonDic["opCode"] = "sweep" # opCode parameter for server

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        received = bytearray()
        try:
            # connect to server
            sock.connect((self.host, self.port))
            # send data
            sock.sendall(json.dumps(jsonDic).encode())
            # receive data back from the server
            # wait for packets until the server is sending them
            while 1:
                tmp = sock.recv(4096)
                if not tmp:
                    break
                received.extend(tmp)
            avg = json.loads(received)
            avg_di = avg["avg_di"]
            avg_dq = avg["avg_dq"]
        finally:
            # shut down
            sock.close()
            # TODO use with syntax

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
                "channel": 1,
                "qubit": pulse.qubit,
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
                "channel": 0,
                "qubit": pulse.qubit,
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
