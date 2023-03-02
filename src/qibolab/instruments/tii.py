""" RFSoC fpga driver.

Supports the following FPGA:
    RFSoC 4x2
"""
import json
import socket

import numpy as np

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
from qibolab.result import ExecutionResults
from qibolab.sweeper import Parameter, Sweeper


class TII_RFSOC4x2(AbstractInstrument):
    """Instrument object for controlling the RFSoC4x2 FPGA.

    The connection requires the FPGA to have a server currently listening.
    The ``connect`` and the ``setup`` functions must be called before playing pulses with
    ``play`` (for arbitrary qibolab ``PulseSequence``) or ``sweep``.

    Args:
        name (str): Name of the instrument instance.
        address (str): IP address and port for connecting to the FPGA.
    """

    def __init__(self, name: str, address: str):
        super().__init__(name, address)
        self.cfg: dict = {}
        self.host, port = address.split(":")
        self.port = int(port)

    def connect(self):
        """Connects to the FPGA instrument."""
        self.is_connected = True

        # Create a socket (SOCK_STREAM means a TCP socket) and send configuration
        self.cfg["opCode"] = "configuration"  # opCode parameter for server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            sock.sendall(json.dumps(self.cfg).encode())

    def setup(self, qubits, repetition_duration, adc_trig_offset, max_gain, sampling_rate, **kwargs):
        """Configures the instrument.

        A connection to the instrument needs to be established beforehand.
        Args: Settings taken from runcard
            qubits: parameter not used
            repetition_duration (int): delay before readout (ms)
            adc_trig_offset (int):
            max_gain (int): defined in dac units so that amplitudes can be relative
            sampling_rate (int): ADC sampling rate

        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """

        if self.is_connected:
            # Load needed settings
            self.cfg = {
                "repetition_duration": repetition_duration,
                "adc_trig_offset": adc_trig_offset,
                "max_gain": max_gain,
                "sampling_rate": sampling_rate
            }

            self.cfg["opCode"] = "setup"  # opCode parameter for server
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.host, self.port))
                sock.sendall(json.dumps(self.cfg).encode())

        else:
            raise Exception("The instrument cannot be set up, there is no connection")

    def play(self, qubits, sequence, relaxation_time, nshots=1000):
        """Executes the sequence of instructions and retrieves the readout results.

        Each readout pulse generates a separate acquisition.

        Args:
            qubits: parameter not used
            sequence (PulseSequence): arbitary qibolab pulse sequence to execute
            nshots (int): number of shots
            relaxation_time (int): delay before readout (ms)

        Returns:
            A dictionary mapping the readout pulse serial to am ExecutionResults object
        """

        json_dic = {}
        json_dic["opCode"] = "execute"
        json_dic["nshots"] = nshots
        json_dic["relaxation_time"] = relaxation_time

        pulses_dic = {}
        for i, pulse in enumerate(sequence):
            pulses_dic[str(i)] = self.convert_pulse_to_dic(pulse)
        json_dic["pulses"] = pulses_dic

        # Create a socket and send pulse sequence to the FPGA
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            sock.sendall(json.dumps(json_dic).encode())
            # read from the server a maximum of 256 bytes (enough for sequence)
            received = sock.recv(256)
            avg = json.loads(received)

            pulse_serial = avg["serial"]
            avgi = np.array(avg["avgi"])
            avgq = np.array(avg["avgq"])

        return {pulse_serial: ExecutionResults.from_components(avgi, avgq)}

    def sweep(self, qubits, sequence, *sweepers, relaxation_time, nshots=1000, average=True):
        """Play a pulse sequence while sweeping one or more parameters.

        Args:
            qubits: parameter not used
            sequence (PulseSequence): arbitary qibolab pulse sequence to execute
            *sweepers (list): A list of qibolab Sweepers objects
            nshots (int): number of shots
            relaxation_time (int): delay before readout (ms)
            average: parameter not used

        Returns:
            A dictionary mapping the readout pulse serial to am ExecutionResults object

        Raises:
            Exception = If attempting to use more than one Sweeper.
            Exception = If average is set to False

        """

        if average is False:
            raise NotImplementedError("Only averaged results are supported")
        if len(sweepers) > 1:
            raise NotImplementedError("Only one sweeper is supported.")

        #  Parsing the sweeper to dictionary and after to a json file
        sweeper = sweepers[0]

        json_dic = {}
        json_dic["nshots"] = nshots
        json_dic["relaxation_time"] = relaxation_time

        json_dic["parameter"] = str(sweeper.parameter)
        start = sweeper.values[0].item()
        expt = len(sweeper.values)
        step = (sweeper.values[1] - sweeper.values[0]).item()
        json_dic["range"] = {"start": start, "step": step, "expt": expt}

        pulses_dic = {}
        for i, pulse in enumerate(sequence.pulses):  # convert pulses to dictionary
            pulses_dic[str(i)] = self.convert_pulse_to_dic(pulse)
        json_dic["pulses"] = pulses_dic

        json_dic["opCode"] = "sweep"  # opCode parameter for server

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            received = bytearray()
            # connect to server
            sock.connect((self.host, self.port))
            # send data
            sock.sendall(json.dumps(json_dic).encode())
            # receive data back from the server
            # wait for packets until the server is sending them
            while 1:
                tmp = sock.recv(4096)
                if not tmp:
                    break
                received.extend(tmp)
            avg = json.loads(received)

            pulse_serial = avg["serial"]
            avgi = np.array(avg["avg_di"])
            avgq = np.array(avg["avg_dq"])

        return {pulse_serial: ExecutionResults.from_components(avgi, avgq)}

    def convert_pulse_to_dic(self, pulse):
        """Funtion to convert pulse object attributes to a dictionary"""
        pulse: Pulse
        pulse_shape: PulseShape

        if pulse.type == PulseType.DRIVE:
            pulse_shape = pulse.shape
            if type(pulse_shape) is Drag:
                shape = "Drag"
                style = "arb"
                rel_sigma = pulse_shape.rel_sigma
                beta = pulse_shape.beta
            elif type(pulse_shape) is Gaussian:
                shape = "Gaussian"
                style = "arb"
                rel_sigma = pulse_shape.rel_sigma
                beta = 0
            elif type(pulse_shape) is Rectangular:
                shape = "Rectangular"
                style = "const"
                rel_sigma = 0
                beta = 0
            pulse_dictionary = {
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
                "serial": pulse.serial,  # TODO remove redundancy
            }

        elif pulse.type == PulseType.READOUT:
            pulse_dictionary = {
                "start": pulse.start,
                "duration": pulse.duration,
                "amplitude": pulse.amplitude,
                "frequency": pulse.frequency,
                "relative_phase": pulse.relative_phase,
                "shape": "const",
                "type": "ro",
                "channel": 0,
                "qubit": pulse.qubit,
                "serial": pulse.serial,  # TODO remove redundancy
            }

        return pulse_dictionary

    def start(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def stop(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def disconnect(self):
        """Closes the connection to the instrument."""
        self.is_connected = False
