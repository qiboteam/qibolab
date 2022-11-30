# -*- coding: utf-8 -*-
import socket
import struct
import time
from bisect import bisect
from typing import List

import numpy as np
from qibo.config import log, raise_error

from qibolab.instruments.abstract import AbstractInstrument
from qibolab.pulses import PulseSequence, PulseType


class PulseBlaster(AbstractInstrument):
    """Driver for the 24-pin PulseBlaster TTL signal generator."""

    def __init__(self, name, address, port=5000):
        super().__init__(name, address)
        self.port = port

    def setup(self, nshots: int, holdtime_ns: int, pins=list(range(24))):
        """Setup the PulseBlaster for the number of shots and the repetition rate.

        Arguments:
            nshots (int): Number of shots
            holdtime_ns (int): Time between TTL pulses
            pins (int[]): Array of pins to be triggered
        """
        p = self.hexify(pins)
        payload = f"setup,{p},{int(nshots)},{int(holdtime_ns)}"
        self.send(payload.encode("utf-8"), True)

    def fire(self):
        """Starts the programmed pulse sequence."""
        self.send(b"fire")

    def stop(self):
        """Stops the PulseBlaster."""
        self.send(b"stop")

    def status(self):
        """Queries the status of the PulseBlaster."""
        return self.send(b"status", True)

    def send(self, command: bytes, expect_reply=False):
        """Sends a command to the PulseBlaster instrument. If a reply is expected, wait for the instrument response.

        Arguments:
            command (bytes): Command and arguments to be sent to the PulseBlaster
            expect_reply (bool): Flag if there is to be a response from the instrument for this command
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.address, self.port))
            s.sendall(command)

            if expect_reply:
                return s.recv(1024).decode("utf-8")

    def start(self):
        pass

    def connect(self):
        pass

    def disconnect(self):
        pass

    @staticmethod
    def hexify(pins: List[int]):
        """Converts a list of pin IDs to hex for the PulseBlaster to trigger.

        Arguments:
            pins (int[]): Array of pins to be triggered

        Returns:
            pins_hex (int): Integer in hex corrresponding to the pins that should be triggered
        """
        return int("".join(["1" if i in set(pins) else "0" for i in reversed(range(24))]), 2)


class IcarusQRFSOC(AbstractInstrument):
    """Driver for the IcarusQ RFSoC socket-based implementation."""

    def __init__(self, name, address, port=8080):

        super().__init__(name, address)
        self.port = port

        self.dac_nchannels = 16
        self.dac_sample_size = 65536
        self.dac_max_volts = 0.3
        self.dac_max_amplitude = 8191
        self.dac_waveform_buffer: np.ndarray = None
        self.nshots: int = None

        # Qibolab unique trackers
        self.last_pulsequence_hash = "uninitialised"
        self.current_pulsesequence_hash = ""
        self.channel_port_map = {}
        self.acquisitons = []
        self.ports = {ch: None for ch in range(self.dac_nchannels)}
        self.channels: list = []
        self.dacs = {}

    def set_adc_sampling_rate(self, adc_sr=None):
        """Sets the sampling rate of all 16 ADCs.

        Arguments:
            adc_sr (float): Sampling rate of the ADC in MHz.
        """

        # Start a connection to the RFSoC board
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.address, self.port))

            # Command 5 tells the RFSoC to set the ADC sampling rate
            cmd = struct.pack("B", 5)
            s.sendall(cmd)

            # Send the sampling rate value
            cmd = struct.pack("d", adc_sr)
            s.sendall(cmd)

            # Get the ADC multi-tile sync return value
            return_mts = struct.unpack("i", s.recv(4))[0]
            if return_mts != 0:
                log.warn("Failled to set ADC Sampling rate properly. Return Value : " + str(return_mts))

    def set_dac_sampling_rate(self, dac_sr=None):
        """Sets the sampling rate of all 16 DACs.

        Arguments:
            dac_sr (float): Sampling rate of the DAC in MHz.
        """

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.address, self.port))
            # Command 7 tells the RFSoC to set the DAC sampling rate
            cmd = struct.pack("B", 7)
            s.sendall(cmd)

            # Send the DAC sampling rate
            cmd = struct.pack("d", dac_sr)
            s.sendall(cmd)

            # Get the DAC multi-tile sync return value
            return_mts = struct.unpack("i", s.recv(4))[0]
            if return_mts != 0:
                log.warn("Failled to set DAC Sampling rate properly. Return Value : " + str(return_mts))

    def get_server_version(self):
        """Fetches the binary version of the board.

        Returns:
            version (string): binary version. <board>-<Release>.<HW>.<SW>.<Commit>
        """

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.address, self.port))

            # Command 9 tells the RFSoC to return the board version information
            s.sendall(struct.pack("B", 9))
            version_str = s.recv(50).decode("utf-8")

        return version_str

    def get_adc_sampling_rate(self):
        """Fetches the sampling rate for the ADCs in MHz

        Returns:
            adc_sr (float): Sampling rate of the ADCs in MHz
        """

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.address, self.port))

            # Command 6 tells the RFSoC to return the ADC sampling rate
            s.sendall(struct.pack("B", 6))
            adc_sr = struct.unpack("d", s.recv(8))[0]

        return adc_sr

    def get_dac_sampling_rate(self):
        """Fetches the sampling rate for the DACs in MHz

        Returns:
            adc_sr (float): Sampling rate of the ADCs in MHz
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.address, self.port))

            # Command 8 tells the RFSoC to return the DAC sampling rate
            s.sendall(struct.pack("B", 8))
            dac_sr = struct.unpack("d", s.recv(8))[0]

        return dac_sr

    @property
    def dac_sampling_rate(self):
        return self.get_dac_sampling_rate()

    @dac_sampling_rate.setter
    def dac_sampling_rate(self, sampling_rate):
        self.set_dac_sampling_rate(dac_sr=sampling_rate)

    @property
    def adc_sampling_rate(self):
        return self.get_adc_sampling_rate()

    @adc_sampling_rate.setter
    def adc_sampling_rate(self, sampling_rate):
        self.set_adc_sampling_rate(adc_sr=sampling_rate)

    def setup(self, dac_sampling_rate: float, adc_sampling_rate: float, channel_port_map: dict, **kwargs):
        """Setup the baord by setting the sampling rate of the DACs and ADCs

        Arguments:
            dac_sampling_rate (float): Sampling rate of the 16 DACs in MHz.
            adc_sampling_rate (float): Sampling rate of the 16 ADCs in MHz.
            channel_dac_map (dict): Mapping of fridge channels to RFSoC DACs
        """

        self.dac_sampling_rate = dac_sampling_rate
        self.adc_sampling_rate = adc_sampling_rate
        self.channel_port_map = channel_port_map
        self.channels = list(self.channel_port_map.keys())
        self.dacs = self.channel_port_map

    def process_pulse_sequence(self, instrument_pulses: PulseSequence, nshots: int, repetition_duration: int):
        """Processes the pulse sequence into np arrays to upload to the board

        Arguments:
            channel_pulses (dict[str, List[Pulse]]): A dictionary of fridge ports mapped to an array of pulses to be sent.
            nshots (int): Number of shots
            repetition_duration (int): Unused parameter for compatiability with other instruments.
        """

        # Update number of shots, even if the pulse sequence is the same
        self.nshots = nshots

        # Check if the new pulse sequence is the same as the currently loaded pulse sequence
        self.current_pulsesequence_hash = hash(instrument_pulses)

        # If they are the same, exit
        if self.current_pulsesequence_hash == self.last_pulsequence_hash:
            return

        # Initialize the waveform buffer
        self.dac_waveform_buffer = np.zeros((self.dac_nchannels, self.dac_sample_size))

        # Reset the acquisition store
        self.acquisitons = []

        # Get the DAC time array
        time_array = 1 / (self.dac_sampling_rate * 1e6) * np.arange(self.dac_sample_size)
        for pulse in instrument_pulses.pulses:

            # Map fridge port to DAC number
            dac = self.channel_port_map[pulse.channel]

            start = pulse.start * 1e-9
            duration = pulse.duration * 1e-9
            end = start + duration

            idx_start = bisect(time_array, start)
            idx_end = bisect(time_array, end)
            t = time_array[idx_start:idx_end]

            # Convert amplitude to DAC bits
            amplitude = pulse.amplitude / self.dac_max_volts * self.dac_max_amplitude
            wfm = amplitude * np.sin(2 * np.pi * pulse.frequency * t + pulse.phase)
            self.dac_waveform_buffer[dac, idx_start:idx_end] += wfm

            # Store the readout pulse
            if pulse.type == PulseType.READOUT:
                self.acquisitons.append((pulse.qubit, pulse.frequency))

    def upload(self):
        """Uploads the pulse sequence to the RFSoC"""

        # Don't upload if the currently loaded pulses are the same as the previous set
        if self.current_pulsesequence_hash == self.last_pulsequence_hash:
            return

        if self.dac_waveform_buffer is None:
            raise_error(RuntimeError, "No pulse sequences currently configured")

        if self.dac_waveform_buffer.shape != (self.dac_nchannels, self.dac_sample_size):
            raise_error(ValueError, "Waveform to be uploaded has invalid size")

        # We shift by the waveform array by 2 bytes as the RFSoC will shift 16 byte data to 14
        payload = (4 * self.dac_waveform_buffer).astype("i2")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.address, self.port))

            # Command 1 tells the RFSoC to await the waveform data to load into DACs
            cmd = struct.pack("B", 1)
            s.sendall(cmd)
            s.sendall(payload.tobytes())

        self.last_pulsequence_hash = self.current_pulsesequence_hash

    def play_sequence(self):
        """DACs are armed on waveform upload, so no method is required to arm the DACs for playback"""
        pass

    def connect(self):
        """Currently we only connect to the board when we have to send a command."""
        # Request the version from the board
        ver = self.get_server_version()
        log.info(f"Connected to {self.name}, version: {ver}")

    def start(self):
        pass

    def stop(self):
        pass

    def disconnect(self):
        pass


class IcarusQ_RFSOC_QRM(IcarusQRFSOC):
    """IcarusQ RFSoC attached with readout capability"""

    def __init__(self, name, address):
        super().__init__(name, address)

        self.pb: PulseBlaster = None

        self.adc_nchannels = 16
        self.adc_sample_size = 65536

        self.qubit_adc_map = {}
        self.adcs_to_read: List[int] = None

    def setup(
        self,
        dac_sampling_rate: float,
        adc_sampling_rate: float,
        channel_port_map: dict,
        qubit_adc_map: dict,
        pulseblaster_address: str,
        pulseblaster_port: int,
        single_shot: bool,
        **kwargs,
    ):
        """Setup the board and assign ADCs to be read. A PulseBlaster is also assigned to trigger this board.

        Arguments:
            dac_sampling_rate (float): Sampling rate of the 16 DACs in MHz.
            adc_sampling_rate (float): Sampling rate of the 16 ADCs in MHz.
            channel_port_map (dict): Mapping of fridge channels to RFSoC DACs
            qubit_adc_map (dict): Mapping of qubit IDs to RFSoC ADCs
            pulseblaster_address (str): IP Address of the attached PulseBlaster.
            pulseblaster_port (int): IP port of the attached PulseBlaster.
            single_shot (bool): Flag to return single shot data.
        """
        super().setup(dac_sampling_rate, adc_sampling_rate, channel_port_map)
        self.qubit_adc_map = qubit_adc_map
        self.adcs_to_read = list(set(list(qubit_adc_map.values())))
        self.pb = PulseBlaster("pb", pulseblaster_address, pulseblaster_port)
        self.single_shot_flag = single_shot

    def play_sequence_and_acquire(self):
        """Starts the experiment sequence and proccesses the results.

        Returns:
            results (dict[int, np.ndarray]): A dictionary of result data for each qubit.
            If single shot is enabled, the result data will be 2D array of data of each shot.
            Else, the result data will be a 1D array of averaged amplitude, phase, I and Q.
        """
        adc_data_buffer = self.acquire()
        results = self.process_adc_data(adc_data_buffer)
        return results

    def acquire(self):
        """Arms the ADC for acquisition, starts the PulseBlaster TTL and acquires the ADC data from the board.

        Returns:
            adc_data_buffer (dict[int, np.ndarray]): A dictionary of ADC data per shot with the ADC channels as the keys.
        """

        num_adcs = len(self.adcs_to_read)
        # Setup the PulseBlaster for the acquisition
        self.pb.setup(self.nshots, holdtime_ns=3e6 * num_adcs)

        adc_data_buffer = {adc: np.zeros((self.nshots, self.adc_sample_size), dtype="i2") for adc in self.adcs_to_read}
        # Each data point of the ADC sample is 16-bit
        BUFFER_SIZE = self.adc_sample_size * 2

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.address, self.port))

            # Command 2 tells the RFSoC to arm the ADCs
            s.sendall(struct.pack("B", 2))
            # Send number of shots as unsigned shot, max 65536, increase in future if needed
            s.sendall(struct.pack("H", self.nshots))
            # Send number of channels
            s.sendall(struct.pack("B", num_adcs))

            for adc in self.adcs_to_read:
                # Send ADC channel number to use
                s.sendall(struct.pack("B", adc))

            # Wait 0.1s for the board to set up the ADC
            time.sleep(0.1)
            self.pb.fire()

            for j in range(self.nshots):
                for k in range(num_adcs):
                    # Each message from the ADC begins with the current shot number and channel number
                    # These messages can be out-of-order due to PS thread execution
                    shotnum = struct.unpack("H", s.recv(2))[0]
                    channel = struct.unpack("H", s.recv(2))[0]

                    r = bytearray()
                    # We start listening for a packet of max BUFFER_SIZE amount of data
                    # However, this packet may have less data than the expected amount of data
                    # Hence, we need to use a while loop to keep iterating until we have received the full amount of data
                    while len(r) < BUFFER_SIZE:
                        packet = s.recv(BUFFER_SIZE - len(r))
                        if packet:
                            r.extend(packet)

                    # Load the data packet and assign it to the internal buffer
                    # Shift incoming 12-bit data from 16-bit to 12-bit
                    adc_data_buffer[channel][shotnum] = np.right_shift(np.frombuffer(r, dtype="i2"), 4)

        self.pb.stop()
        return adc_data_buffer

    def process_adc_data(self, adc_data_buffer: dict):
        """Processes the ADC data into amplitude, phase, I and Q for the readout signals.

        Arguments:
            adc_data_buffer (dict[int, np.ndarray]): A dictionary of ADC data per shot with the ADC channels as the keys.

        Returns:
            results (dict[int, np.ndarray]): A dictionary of result data for each qubit.
            If single shot is enabled, the result data will be 2D array of data of each shot.
            Else, the result data will be a 1D array of averaged amplitude, phase, I and Q.
        """

        result = {}
        time_array = 1 / (self.adc_sampling_rate * 1e6) * np.arange(self.adc_sample_size)

        for qubit, readout_frequency in self.acquisitons:
            adc = self.qubit_adc_map[qubit]
            data = adc_data_buffer[adc]

            cos = np.cos(2 * np.pi * readout_frequency * time_array)
            sin = np.sin(2 * np.pi * readout_frequency * time_array)

            if self.single_shot_flag:
                res = np.zeros((self.nshots, 4))

                for shotnum, signal in data.items():
                    I = np.sum(signal * sin)
                    Q = np.sum(signal * cos)
                    amplitude = np.sqrt(I**2 + Q**2)
                    phase = np.arctan2(Q, I)

                    res[shotnum] = amplitude, phase, I, Q

                result[qubit] = res

            else:
                signal = np.average(data, axis=0)
                I = np.sum(signal * sin)
                Q = np.sum(signal * cos)
                amplitude = np.sqrt(I**2 + Q**2)
                phase = np.arctan2(Q, I)

                result[qubit] = np.array([amplitude, phase, I, Q])

        return result


class IcarusQ_RFSOC_QCM(IcarusQRFSOC):
    pass
