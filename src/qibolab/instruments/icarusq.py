from bisect import bisect
from typing import List

import numpy as np
from qibo.config import raise_error

from qibolab.instruments.abstract import Instrument, InstrumentException
from qibolab.pulses import Pulse


class TektronixAWG5204(Instrument):
    def __init__(self, name, address):
        super().__init__(name, address)
        # Phase offset for each channel for IQ sideband optimziation
        self.channel_phase: "list[float]" = []
        # Time buffer at the start and end of the pulse sequence to ensure that the minimum samples of the instrument are reached
        self.pulse_buffer: float = 1e-6
        self.device = None
        self.sample_rate = None

    rw_property_wrapper = lambda parameter: property(
        lambda self: self.device.get(parameter),
        lambda self, x: self.device.set(parameter, x),
    )
    rw_property_wrapper("sample_rate")

    def connect(self):
        if not self.is_connected:
            from qcodes.instrument_drivers.tektronix.AWG70000A import AWG70000A

            try:
                self.device = AWG70000A(self.name, self.address, num_channels=4)
            except Exception as exc:
                raise InstrumentException(self, str(exc))
            self.is_connected = True
        else:
            raise_error(
                Exception, "There is an open connection to the instrument already"
            )

    def setup(self, **kwargs):
        if self.is_connected:
            # Set AWG to external reference, 10 MHz
            self.device.write("CLOC:SOUR EFIX")
            # Set external trigger to 1V
            self.device.write("TRIG:LEV 1")
            self.sample_rate = kwargs.pop("sample_rate")

            resolution = kwargs.pop("resolution")
            amplitude = kwargs.pop("amplitude")
            offset = kwargs.pop("offset")

            for idx, channel in enumerate(range(1, self.device.num_channels + 1)):
                awg_ch = getattr(self.device, f"ch{channel}")
                awg_ch.awg_amplitude(amplitude[idx])
                awg_ch.resolution(resolution)
                self.device.write(
                    f"SOURCE{channel}:VOLTAGE:LEVEL:IMMEDIATE:OFFSET {offset[idx]}"
                )

            self.__dict__.update(kwargs)
        else:
            raise_error(Exception, "There is no connection to the instrument")

    def generate_waveforms_from_pulse(self, pulse: Pulse, time_array: np.ndarray):
        """Generates a numpy array based on the pulse parameters.

        Arguments:
            pulse (qibolab.pulses.Pulse | qibolab.pulses.ReadoutPulse): Pulse to be compiled
            time_array (numpy.ndarray): Array corresponding to the global time
        """
        i_ch, q_ch = pulse.channel

        i = pulse.envelope_i * np.cos(
            2 * np.pi * pulse.frequency * time_array
            + pulse.phase
            + self.channel_phase[i_ch]
        )
        q = (
            -1
            * pulse.envelope_i
            * np.sin(
                2 * np.pi * pulse.frequency * time_array
                + pulse.phase
                + self.channel_phase[q_ch]
            )
        )
        return i, q

    def translate(self, sequence: List[Pulse], nshots=None):
        """Translates the pulse sequence into a numpy array.

        Arguments:
            sequence (qibolab.pulses.Pulse[]): Array containing pulses to be fired on this instrument.
            nshots (int): Number of repetitions.
        """

        # First create np arrays for each channel
        start = min(pulse.start for pulse in sequence)
        end = max(pulse.start + pulse.duration for pulse in sequence)
        time_array = np.arange(
            start * 1e-9 - self.pulse_buffer,
            end * 1e-9 + self.pulse_buffer,
            1 / self.sample_rate,
        )
        waveform_arrays = np.zeros((self.device.num_channels, len(time_array)))

        for pulse in sequence:
            start_index = bisect(time_array, pulse.start * 1e-9)
            end_index = bisect(time_array, (pulse.start + pulse.duration) * 1e-9)
            i_ch, q_ch = pulse.channel
            i, q = self.generate_waveforms_from_pulse(
                pulse, time_array[start_index:end_index]
            )
            waveform_arrays[i_ch, start_index:end_index] += i
            waveform_arrays[q_ch, start_index:end_index] += q

        return waveform_arrays

    def upload(self, waveform: np.ndarray):
        """Uploads a nchannels X nsamples array to the AWG, load it into memory
        and assign it to the channels for playback."""

        # TODO: Add additional check to ensure all waveforms are of the same size? Should be caught by qcodes driver anyway.
        if len(waveform) != self.device.num_channels:
            raise_error(Exception, "Invalid waveform given")

        # Clear existing waveforms in memory
        self.device.write("WLIS:WAV:DEL ALL")

        # Upload waveform, load into memory and assign to each channel
        for idx, channel in enumerate(range(1, self.device.num_channels + 1)):
            awg_ch = getattr(self.device, f"ch{channel}")
            wfmx = self.device.makeWFMXFile(waveform[idx], awg_ch.awg_amplitude())
            self.device.sendWFMXFile(wfmx, f"ch{channel}.wfmx")
            self.device.loadWFMXFile(f"ch{channel}.wfmx")
            self.device.write(f'SOURce{channel}:CASSet:WAVeform "ch{channel}"')

    def start(self):
        pass

    def play_sequence(self):
        for channel in range(1, self.device.num_channels + 1):
            awg_ch = getattr(self.device, f"ch{channel}")
            awg_ch.state(1)
            self.device.write(f"SOURce{channel}:RMODe TRIGgered")
            self.device.write(f"SOURce{channel}:TINPut ATR")
        self.device.play()
        self.device.wait_for_operation_to_complete()

    def stop(self):
        self.device.stop()
        for channel in range(1, self.device.num_channels + 1):
            awg_ch = getattr(self.device, f"ch{channel}")
            awg_ch.state(0)
        self.device.wait_for_operation_to_complete()

    def disconnect(self):
        if self.is_connected:
            self.device.stop()
            self.device.close()
            self.is_connected = False

    def __del__(self):
        self.disconnect()

    def close(self):
        if self.is_connected:
            self.stop()
            self.device.close()
            self.is_connected = False


class MCAttenuator(Instrument):
    """Driver for the MiniCircuit RCDAT-8000-30 variable attenuator."""

    def connect(self):
        pass

    def setup(self, attenuation: float, **kwargs):
        """Assigns the attenuation level on the attenuator.

        Arguments:
            attenuation(float
            ): Attenuation setting in dB. Ranges from 0 to 35.
        """
        import urllib3

        http = urllib3.PoolManager()
        http.request("GET", f"http://{self.address}/SETATT={attenuation}")

    def start(self):
        pass

    def stop(self):
        pass

    def disconnect(self):
        pass


class QuicSyn(Instrument):
    """Driver for the National Instrument QuicSyn Lite local oscillator."""

    def connect(self):
        if not self.is_connected:
            import pyvisa as visa

            rm = visa.ResourceManager()
            try:
                self.device = rm.open_resource(self.address)
            except Exception as exc:
                raise InstrumentException(self, str(exc))
            self.is_connected = True

    def setup(self, frequency: float, **kwargs):
        """Sets the frequency in Hz."""
        if self.is_connected:
            self.device.write("0601")
            self.frequency(frequency)

    def frequency(self, frequency):
        self.device.write("FREQ {:f}Hz".format(frequency))

    def start(self):
        """Starts the instrument."""
        self.device.write("0F01")

    def stop(self):
        """Stops the instrument."""
        self.device.write("0F00")

    def __del__(self):
        self.disconnect()

    def disconnect(self):
        if self.is_connected:
            self.stop()
            self.device.close()
            self.is_connected = False


class AlazarADC(Instrument):
    """Driver for the AlazarTech ATS9371 ADC."""

    def __init__(self, name, address):
        super().__init__(name, address)
        self.controller = None
        self.device = None

    def connect(self):
        if not self.is_connected:
            from qcodes.instrument_drivers.AlazarTech.AlazarADC import (  # pylint: disable=E0401, E0611
                ADCController,
            )
            from qcodes.instrument_drivers.AlazarTech.ATS9371 import (  # pylint: disable=E0401, E0611
                AlazarTech_ATS9371,
            )

            try:
                self.device = AlazarTech_ATS9371(self.address)
                self.controller = ADCController(self.name, self.address)
            except Exception as exc:
                raise InstrumentException(self, str(exc))
            self.is_connected = True

    def setup(self, trigger_volts, **kwargs):
        """Sets the frequency in Hz."""
        if self.is_connected:
            input_range_volts = 2.5
            trigger_level_code = int(128 + 127 * trigger_volts / input_range_volts)
            with self.device.syncing():
                self.device.clock_source("EXTERNAL_CLOCK_10MHz_REF")
                self.device.external_sample_rate(1_000_000_000)
                self.device.clock_edge("CLOCK_EDGE_RISING")
                self.device.decimation(1)
                self.device.coupling1("DC")
                self.device.coupling2("DC")
                self.device.channel_range1(0.02)
                self.device.channel_range2(0.02)
                self.device.impedance1(50)
                self.device.impedance2(50)
                self.device.bwlimit1("DISABLED")
                self.device.bwlimit2("DISABLED")
                self.device.trigger_operation("TRIG_ENGINE_OP_J")
                self.device.trigger_engine1("TRIG_ENGINE_J")
                self.device.trigger_source1("EXTERNAL")
                self.device.trigger_slope1("TRIG_SLOPE_POSITIVE")
                self.device.trigger_level1(trigger_level_code)
                self.device.trigger_engine2("TRIG_ENGINE_K")
                self.device.trigger_source2("DISABLE")
                self.device.trigger_slope2("TRIG_SLOPE_POSITIVE")
                self.device.trigger_level2(128)
                self.device.external_trigger_coupling("DC")
                self.device.external_trigger_range("ETR_2V5")
                self.device.trigger_delay(0)
                self.device.timeout_ticks(0)

            samples = kwargs.pop("samples")
            self.controller.samples = samples
            self.__dict__.update(kwargs)

    def arm(self, nshots, readout_start):
        with self.device.syncing():
            self.device.trigger_delay(
                int(int((readout_start * 1e-9 + 4e-6) / 1e-9 / 8) * 8)
            )
        self.controller.arm(nshots)

    def play_sequence_and_acquire(self):
        """This method performs an acquisition, which is the get_cmd for the
        acquisiion parameter of this instrument :return:"""
        raw = self.device.acquire(
            acquisition_controller=self.controller, **self.controller.acquisitionkwargs
        )
        return self.process_result(raw)

    def process_result(self, readout_frequency=100e6, readout_channels=[0, 1]):
        """Returns the processed signal result from the ADC.

        Arguments:
            readout_frequency (float): Frequency to be used for signal processing.
            readout_channels (int[]): Channels to be used for signal processing.

        Returns:
            ampl (float): Amplitude of the processed signal.
            phase (float): Phase shift of the processed signal in degrees.
            it (float): I component of the processed signal.
            qt (float): Q component of the processed signal.
        """

        input_vec_I = self.device._processed_data[readout_channels[0]]
        input_vec_Q = self.device._processed_data[readout_channels[1]]
        it = 0
        qt = 0
        for i in range(self.device.samples_per_record):
            it += input_vec_I[i] * np.cos(
                2 * np.pi * readout_frequency * self.device.time_array[i]
            )
            qt += input_vec_Q[i] * np.cos(
                2 * np.pi * readout_frequency * self.device.time_array[i]
            )
        phase = np.arctan2(qt, it)
        ampl = np.sqrt(it**2 + qt**2)

        return ampl, phase, it, qt

    def start(self):
        """Starts the instrument."""

    def stop(self):
        """Stops the instrument."""

    def disconnect(self):
        if self.is_connected:
            self.device.close()
            self.controller.close()
