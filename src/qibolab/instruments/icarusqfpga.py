import bisect
import socket
import struct
import threading
import time
from typing import List

import numpy as np

from qibolab.instruments.abstract import Instrument
from qibolab.pulses import Pulse


class PulseBlaster(Instrument):
    """Driver for the 24-pin PulseBlaster TTL signal generator."""

    def __init__(self, name, address, port=5000):
        super().__init__(name, address)
        self.port = port
        self._pins = None

    def setup(self, holdtime_ns, pins=None, **kwargs):
        """Setup the PulseBlaster.

        Arguments:
            holdtime_ns (int): TTL pulse length and delay between TTL pulses. The experiment repetition frequency is 1 / (2 * holdtime_ns).
            pins (int): Pins to trigger in hex, defaults to all pins.
        """
        if pins is None:
            pins = list(range(24))
        self._pins = self._hexify(pins)
        self._holdtime = holdtime_ns

    def arm(self, nshots, readout_start=0):
        """Arm the PulseBlaster for playback. Sends a signal to the instrument
        to setup the pulse sequence and repetition.

        Arguments:
            nshots (int): Number of TTL triggers to repeat.
        """
        payload = f"setup,{self._pins},{nshots},{self._holdtime}"
        return self._send(payload.encode("utf-8"), True)

    def fire(self):
        """Starts the PulseBlaster."""
        self._send(b"fire")

    def start(self):
        pass

    def stop(self):
        self._send(b"stop")

    def status(self):
        return self._send(b"status", True)

    def _send(self, payload, retval=False):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.address, self.port))
            s.sendall(payload)

            if retval:
                return s.recv(1024).decode("utf-8")

    def play_sequence(self):
        self.fire()

    def connect(self):
        pass

    def disconnect(self):
        pass

    @staticmethod
    def _hexify(pins):
        return int(
            "".join(["1" if i in set(pins) else "0" for i in reversed(range(24))]), 2
        )


class IcarusQFPGA(Instrument):
    """Driver for the IcarusQ RFSoC socket-based implementation."""

    def __init__(self, name, address, port=8080):
        super().__init__(name, address)
        self._dac_sample_size = 65536
        self._adc_sample_size = 65536
        self._dac_nchannels = 16
        self._adc_nchannels = 16
        self._dac_sampling_rate = 5898240000
        self._adc_sampling_rate = 1966080000
        self._nshots = 0

        self.port = port
        self._thread = None
        self._buffer = None
        self._adcs_to_read = None
        self.nshots = None

    def setup(self, dac_sampling_rate, adcs_to_read, **kwargs):
        """Sets the sampling rate of the RFSoC. May need to be repeated several
        times due to multi-tile sync error.

        Arguments:
            dac_sampling_rate_id (int): Sampling rate ID to be set on the RFSoC.
            dac_sampling_rate_6g_id (int): Optional sampling rate ID for the 6GS/s mode if used.
        """
        self._adcs_to_read = adcs_to_read
        self._dac_sampling_rate = dac_sampling_rate

    def translate(self, sequence: List[Pulse], nshots):
        """Translates the pulse sequence into a numpy array."""

        # Waveform is 14-bit resolution on the DACs, so we first create 16-bit arrays to store the data.
        waveform = np.zeros((self._dac_nchannels, self._dac_sample_size), dtype="i2")

        # The global time can first be set as float to handle rounding errors.
        time_array = (
            1 / self._dac_sampling_rate * np.arange(0, self._dac_sample_size, 1)
        )

        for pulse in sequence:
            # Get array indices corresponding to the start and end of the pulse. Note that the pulse time parameters are in ns and require conversion.
            start = bisect.bisect(time_array, pulse.start * 1e-9)
            end = bisect.bisect(time_array, (pulse.start + pulse.duration) * 1e-9)

            # Create the pulse waveform and cast it to 16-bit. The ampltiude is max signed 14-bit (+- 8191) and the indices should take care of any overlap of pulses.
            # 2-byte bit shift for downsampling from 16 bit to 14 bit
            pulse_waveform = (
                4
                * np.sin(
                    2 * np.pi * pulse.frequency * time_array[start:end] + pulse.phase
                )
            ).astype("i2")
            waveform[pulse.channel, start:end] += pulse_waveform

        self.nshots = nshots

        return waveform

    def upload(self, waveform):
        """Uploads a numpy array of size DAC_CHANNELS X DAC_SAMPLE_SIZE to the
        PL memory.

        Arguments:
            waveform (numpy.ndarray): Numpy array of size DAC_CHANNELS X DAC_SAMPLE_SIZE with type signed short.
        """

        # TODO: Implement checks for size and dtype of waveform.

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.address, self.port))
            # Signal to the RFSoC to start listening for DAC waveforms and load into PL mem.
            s.sendall(struct.pack("B", 1))
            s.sendall(waveform.tobytes())

    def play_sequence(self):
        """DACs are automatically armed for playbacked when waveforms are
        loaded, no need to signal."""
        self._buffer = np.zeros((self._adc_nchannels, self._adc_sample_size))
        self._thread = threading.Thread(target=self._play, args=(self.nshots,))
        self._thread.start()

        time.sleep(0.1)  # Use threading lock and socket signals instead of hard sleep?

    def play_sequence_and_acquire(self):
        """Signal the RFSoC to arm the ADC and start data transfer into PS
        memory. Starts a thread to listen for ADC data from the RFSoC.

        Arguments:
            nshots (int): Number of shots.
        """
        # Create buffer to hold ADC data.
        # TODO: Create flag to handle single shot measurement / buffer assignment per shot.

    def _play(self, nshots):
        """Starts ADC data acquisition and transfer on the RFSoC."""

        if len(self._adcs_to_read) == 0:
            return

        # 2 bytes per ADC data point
        BUF_SIZE = int(self._adc_sample_size * 2)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.address, self.port))
            # Signal RFSoC to arm ADC and expect `nshots` number of triggers.
            s.sendall(struct.pack("B", 2))
            s.sendall(struct.pack("H", nshots))
            s.sendall(
                struct.pack("B", len(self._adcs_to_read))
            )  # send number of channels

            for adc in self._adcs_to_read:
                s.sendall(struct.pack("B", adc))  # send ADC channel to read

            # Use the same socket to start listening for ADC data transfer.
            for _ in range(nshots):
                for _ in self._adcs_to_read:
                    # IcarusQ board may send channel/shot data out of order due to threading implementation.
                    # shotnum = struct.unpack("H", s.recv(2))[0]
                    channel = struct.unpack("H", s.recv(2))[0]

                    # Start listening for ADC_SAMPLE_SIZE * 2 bytes corresponding to data per channel.
                    r = bytearray()
                    while len(r) < BUF_SIZE:
                        # Socket implementation does not return exactly desired amount of bytes, keep querying until bytearray reaches expected amount of bytes.
                        # TODO: Look for `MSG_WAITALL` flag in socket recv.
                        packet = s.recv(BUF_SIZE - len(r))
                        if packet:
                            r.extend(packet)

                    # Accumulate ADC data in buffer, buffer float dtype should be enough to prevent overflow.
                    self._buffer[channel] += np.frombuffer(r, dtype="i2")

        # Average buffer at the end of measurement
        self._buffer = self._buffer / nshots

    def result(self, readout_frequency, readout_channel):
        """Returns the processed signal result from the ADC.

        Arguments:
            readout_frequency (float): Frequency to be used for signal processing.
            readout_channel (int): Channel to be used for signal processing.

        Returns:
            ampl (float): Amplitude of the processed signal.
            phase (float): Phase shift of the processed signal in degrees.
            it (float): I component of the processed signal.
            qt (float): Q component of the processed signal.
        """

        # Wait for ADC data acquisition to complete
        self._thread.join()

        input_vec = self._buffer[readout_channel]

        time_vec = 1 / self._adc_sampling_rate * np.arange(0, self._adc_sample_size, 1)
        vec_I = np.sin(2 * np.pi * readout_frequency * time_vec)
        vec_Q = np.cos(2 * np.pi * readout_frequency * time_vec)

        it = np.sum(vec_I * input_vec)
        qt = np.sum(vec_Q * input_vec)
        phase = np.arctan2(qt, it)
        ampl = np.sqrt(it**2 + qt**2)

        return ampl, phase, it, qt

    def start(self):
        pass

    def connect(self):
        pass

    def stop(self):
        pass

    def disconnect(self):
        pass
