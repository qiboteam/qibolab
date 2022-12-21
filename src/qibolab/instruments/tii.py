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
class tii_rfsoc4x2(AbstractInstrument):
    #
    #################################################################
    def __init__(self, name: str, address: str): #, setting_parameters: dict):
        super().__init__(name, address)
        #self.device: QbloxQrmQcm = None
        self.cfg: dict = {}
        self.host: str
        self.port: str
        self.host, port = address.split(":")
        self.port = int(port)


        """
        self.ports: dict = {}
        self.acquisition_hold_off: int
        self.acquisition_duration: int
        self.discretization_threshold_acq: float
        self.phase_rotation_acq: float
        self.channel_port_map: dict = {}
        self.channels: list = []

        #self._cluster: QbloxCluster = None
        self._input_ports_keys = ["i1"]
        self._output_ports_keys = ["o1"]
        self._sequencers: dict[Sequencer] = {"o1": []}
        self._port_channel_map: dict = {}
        self._last_pulsequence_hash: int = 0
        self._current_pulsesequence_hash: int
        self._device_parameters = {}
        self._device_num_output_ports = 1
        self._device_num_sequencers: int
        self._free_sequencers_numbers: list[int] = []
        self._used_sequencers_numbers: list[int] = []
        self._unused_sequencers_numbers: list[int] = []
        """

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

    def setup(self, **kwargs):
        """Configures the instrument.

        A connection to the instrument needs to be established beforehand.
        Args:
            **kwargs: dict = A dictionary of settings loaded from the runcard:
                kwargs['channel_port_map']
                kwargs['ports']['o1']['attenuation']
                kwargs['ports']['o1']['lo_enabled']
                kwargs['ports']['o1']['lo_frequency']
                kwargs['ports']['o1']['gain']
                kwargs['ports']['o1']['hardware_mod_en']
                kwargs['ports']['i1']['hardware_demod_en']
                kwargs['acquisition_hold_off']
                kwargs['acquisition_duration']
        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """

        if self.is_connected:
            # Load settings
            self.cfg = kwargs 
            jsonDic = self.cfg
            jsonDic["opCode"] = "setup"
            # Create a socket (SOCK_STREAM means a TCP socket)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                # Connect to server and send data
                sock.connect((self.host, self.port))
                sock.sendall(json.dumps(jsonDic).encode())

            sock.close()

        else:
            raise Exception("The instrument cannot be set up, there is no connection")

        """
        self.experiment = self.settings["instruments"]["tii_rfsoc4x2"]["experiment"]

        jsonDic = self.experiment
        jsonDic["opCode"] = "setup"
        # Create a socket (SOCK_STREAM means a TCP socket)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to server and send data
            sock.connect((self.host, self.port))
            sock.sendall(json.dumps(jsonDic).encode())

        sock.close()
        """

    def play_sequence_and_acquire(self, sequence):
        """Executes the sequence of instructions and retrieves the readout results.

        Each readout pulse generates a separate acquisition.
        Returns:
            Two array with real and imaginary parts if i and q (already averages)

        """
        ps: PulseShape

        jsonDic = {}
        i = 0
        shape = "const"
        for pulse in sequence:
            if pulse.channel == 1:   
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

                pulseDic = {
                    "start": pulse.start,
                    "duration": pulse.duration,
                    "amplitude": pulse.amplitude,
                    "frequency": pulse.frequency,
                    "relative_phase": pulse.relative_phase,
                    "shape": shape,
                    "style": style,
                    "rel_sigma": rel_sigma,
                    "beta": beta,
                    "channel": pulse.channel,
                    #                        "type": pulse.type,
                    "qubit": pulse.qubit,
                }
                jsonDic["pulse" + str(i)] = pulseDic
                i = i + 1
        for pulse in sequence:
            if pulse.channel == 0:    
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

                pulseDic = {
                    "start": pulse.start,
                    "duration": pulse.duration,
                    "amplitude": pulse.amplitude,
                    "frequency": pulse.frequency,
                    "relative_phase": pulse.relative_phase,
                    "shape": shape,
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
            avgi = avg["avgi"]
            avgq = avg["avgq"]
        sock.close()
        return avgi, avgq





    def start(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass



    def stop(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def disconnect(self):
        """Closes the connection to the instrument."""
        self.is_connected = False
