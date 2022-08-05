import time
import numpy as np
from typing import List
from bisect import bisect
from qibo.config import raise_error
from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.pulses import Pulse

class TektronixAWG5204(AbstractInstrument):

    def __init__(self, name, address):
        super().__init__(name, address)
        # Phase offset for each channel for IQ sideband optimziation
        self.channel_phase: "list[float]" = []
        # Time buffer at the start and end of the pulse sequence to ensure that the minimum samples of the instrument are reached
        self.pulse_buffer: float = 1e-6

    rw_property_wrapper = lambda parameter: property(lambda self: self.device.get(parameter), lambda self,x: self.device.set(parameter,x))
    rw_property_wrapper('sample_rate')


    def connect(self):
        if not self.is_connected:
            from qcodes.instrument_drivers.tektronix.AWG70000A import AWG70000A
            try:
                self.device = AWG70000A(self.name, self.address, num_channels=4)
            except Exception as exc:
                raise InstrumentException(self, str(exc))
            self.is_connected = True
        else:
            raise_error(Exception,'There is an open connection to the instrument already')

    def setup(self, **kwargs):
        if self.is_connected:
            # Set AWG to external reference, 10 MHz
            self.device.write("CLOC:SOUR EFIX")
            # Set external trigger to 1V
            self.device.write('TRIG:LEV 1')
            self.sample_rate = kwargs.pop('sample_rate')

            resolution = kwargs.pop('resolution')
            amplitude = kwargs.pop('amplitude')
            offset = kwargs.pop('offset')

            for idx, channel in enumerate(range(1, self.device.num_channels + 1)):
                awg_ch = getattr(self.device, f"ch{channel}")
                awg_ch.awg_amplitude(amplitude[idx])
                awg_ch.resolution(resolution)
                self.device.write(f"SOURCE{channel}:VOLTAGE:LEVEL:IMMEDIATE:OFFSET {offset[idx]}")

            self.__dict__.update(kwargs)
        else:
            raise_error(Exception,'There is no connection to the instrument')

    def generate_waveforms_from_pulse(self, pulse: Pulse, time_array: np.ndarray):
        """Generates a numpy array based on the pulse parameters
        
        Arguments:
            pulse (qibolab.pulses.Pulse | qibolab.pulses.ReadoutPulse): Pulse to be compiled
            time_array (numpy.ndarray): Array corresponding to the global time
        """
        i_ch, q_ch = pulse.channel

        i = pulse.envelope_i * np.cos(2 * np.pi * pulse.frequency * time_array + pulse.phase + self.channel_phase[i_ch])
        q = -1 * pulse.envelope_i * np.sin(2 * np.pi * pulse.frequency * time_array + pulse.phase + self.channel_phase[q_ch])
        return i, q


    def translate(self, sequence: List[Pulse], nshots=None):
        """
        Translates the pulse sequence into a numpy array.

        Arguments:
            sequence (qibolab.pulses.Pulse[]): Array containing pulses to be fired on this instrument.
            nshots (int): Number of repetitions.
        """

        # First create np arrays for each channel
        start = min(pulse.start for pulse in sequence)
        end = max(pulse.start + pulse.duration for pulse in sequence)
        time_array = np.arange(start * 1e-9 - self.pulse_buffer, end * 1e-9 + self.pulse_buffer, 1 / self.sample_rate)
        waveform_arrays = np.zeros((self.device.num_channels, len(time_array)))

        for pulse in sequence:
            start_index = bisect(time_array, pulse.start * 1e-9)
            end_index = bisect(time_array, (pulse.start + pulse.duration) * 1e-9)
            i_ch, q_ch = pulse.channel
            i, q = self.generate_waveforms_from_pulse(pulse, time_array[start_index:end_index])
            waveform_arrays[i_ch, start_index:end_index] += i
            waveform_arrays[q_ch, start_index:end_index] += q

        return waveform_arrays

    def upload(self, waveform: np.ndarray):
        """Uploads a nchannels X nsamples array to the AWG, load it into memory and assign it to the channels for playback.
        """

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
            self.device.write(f'SOURce{channel}:RMODe TRIGgered')
            self.device.write(f'SOURce{channel}:TINPut ATR')
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

class MCAttenuator(AbstractInstrument):
    """Driver for the MiniCircuit RCDAT-8000-30 variable attenuator.
    """

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
        http.request('GET', f'http://{self.address}/SETATT={attenuation}')

    def start(self):
        pass

    def stop(self):
        pass
        
    def disconnect(self):
        pass

class QuicSyn(AbstractInstrument):
    """Driver for the National Instrument QuicSyn Lite local oscillator.
    """

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
        """
        Sets the frequency in Hz
        """
        if self.is_connected:
            self.device.write('0601')
            self.frequency(frequency)

    def frequency(self, frequency):
        self.device.write('FREQ {0:f}Hz'.format(frequency))

    def start(self):
        """Starts the instrument.
        """
        self.device.write('0F01')

    def stop(self):
        """Stops the instrument.
        """
        self.device.write('0F00')

    def __del__(self):
        self.disconnect()

    def disconnect(self):
        if self.is_connected:
            self.stop()
            self.device.close()
            self.is_connected = False

class AlazarADC(AbstractInstrument):
    """Driver for the AlazarTech ATS9371 ADC.
    """
    def __init__(self, name, address):
        super().__init__(name, address)
        self.controller = None

    def connect(self):
        if not self.is_connected:
            from qcodes.instrument_drivers.AlazarTech.ATS9371 import AlazarTech_ATS9371 # pylint: disable=E0401, E0611
            from qcodes.instrument_drivers.AlazarTech.AlazarADC import ADCController # pylint: disable=E0401, E0611
            try:
                self.device = AlazarTech_ATS9371(self.address)
                self.controller = ADCController(self.name, self.address)
            except Exception as exc:
                raise InstrumentException(self, str(exc))
            self.is_connected = True

    def setup(self, trigger_volts, **kwargs):
        """
        Sets the frequency in Hz
        """
        if self.is_connected:
            input_range_volts = 2.5
            trigger_level_code = int(128 + 127 * trigger_volts / input_range_volts)
            with self.device.syncing():
                self.device.clock_source("EXTERNAL_CLOCK_10MHz_REF")
                self.device.external_sample_rate(1_000_000_000)
                self.device.clock_edge("CLOCK_EDGE_RISING")
                self.device.decimation(1)
                self.device.coupling1('DC')
                self.device.coupling2('DC')
                self.device.channel_range1(.02)
                self.device.channel_range2(.02)
                self.device.impedance1(50)
                self.device.impedance2(50)
                self.device.bwlimit1("DISABLED")
                self.device.bwlimit2("DISABLED")
                self.device.trigger_operation('TRIG_ENGINE_OP_J')
                self.device.trigger_engine1('TRIG_ENGINE_J')
                self.device.trigger_source1('EXTERNAL')
                self.device.trigger_slope1('TRIG_SLOPE_POSITIVE')
                self.device.trigger_level1(trigger_level_code)
                self.device.trigger_engine2('TRIG_ENGINE_K')
                self.device.trigger_source2('DISABLE')
                self.device.trigger_slope2('TRIG_SLOPE_POSITIVE')
                self.device.trigger_level2(128)
                self.device.external_trigger_coupling('DC')
                self.device.external_trigger_range('ETR_2V5')
                self.device.trigger_delay(0)
                self.device.timeout_ticks(0)

            samples = kwargs.pop("samples")
            self.controller.samples = samples
            self.__dict__.update(kwargs)

    def arm(self, nshots, readout_start):
        with self.device.syncing():
            self.device.trigger_delay(int(int((readout_start * 1e-9 + 4e-6) / 1e-9 / 8) * 8))
        self.controller.arm(nshots)
        
    def acquire(self):
        """
        this method performs an acquisition, which is the get_cmd for the
        acquisiion parameter of this instrument
        :return:
        """
        raw = self.device.acquire(acquisition_controller=self.controller, **self.controller.acquisitionkwargs)
        return raw

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

        input_vec_I = self.controller._processed_data[readout_channels[0]]
        input_vec_Q = self.controller._processed_data[readout_channels[1]]
        it = 0
        qt = 0
        for i in range(self.controller.samples_per_record):
            it += input_vec_I[i] * np.cos(2 * np.pi * readout_frequency * self.controller.time_array[i])
            qt += input_vec_Q[i] * np.cos(2 * np.pi * readout_frequency * self.controller.time_array[i])
        phase = np.arctan2(qt, it)
        ampl = np.sqrt(it**2 + qt**2)
        
        return ampl, phase, it, qt

    def start(self):
        """Starts the instrument.
        """
        pass

    def stop(self):
        """Stops the instrument.
        """
        pass

    def disconnect(self):
        if self.is_connected:
            self.device.close()
            self.controller.close()

class IcarusQRack_QRM(AbstractInstrument):
    """Rack system using the Tektronix AWG5204 and the AlazarTech ATS9371.
    """
    def __init__(self, name, address):
        super().__init__(name, address)

        self.awg = TektronixAWG5204(f"{name}_awg", address)
        self.adc = AlazarADC(f"{name}_adc", address)
        
        self.awg_waveform_buffer = None
        self.nshots = None
        self.readout_start = None

        # Qibolab unique trackers
        self.last_pulsequence_hash = "uninitialised"
        self.current_pulsesequence_hash = ""
        self.channel_port_map = {}
        self.acquisitons = []

    def connect(self):
        """Connects to the AWG and ADC.
        """
        if not self.is_connected:
            self.awg.connect()
            self.adc.connect()
            self.is_connected = True
        else:
            raise_error(Exception,'There is an open connection to the instrument already')

    def setup(self,
              awg_settings: dict,
              adc_settings: dict,
              channel_port_map: dict,
              **kwargs):
        """Setup the AWG and the ADC and assign the fridge port to AWG channel mapping.

        Arguments:
            awg_settings (dict): @see TektronixAWG5204.setup for more information
            adc_settings(dict): @see AlazarADC.setup for more information
            channel_port_map (dict): Dictionary mapping fridge ports to AWG channels
        """
        self.awg.setup(**awg_settings)
        self.adc.setup(**adc_settings)
        self.channel_port_map = channel_port_map

    def process_pulse_sequence(self, channel_pulses: "dict[str, List[Pulse]]", nshots: int):
        """Processes the pulse sequence into np arrays to upload to the AWG

        Arguments:
            channel_pulses (dict[str, List[Pulse]]): A dictionary of fridge ports mapped to an array of pulses to be sent.
            nshots (int): Number of shots
        """

        # Update number of shots, even if the pulse sequence is the same
        self.nshots = nshots

        # Check if the new pulse sequence is the same as the currently loaded pulse sequence
        self.current_pulsesequence_hash = ""
        for channel_sequence in channel_pulses.values():
            for pulse in channel_sequence:
                self.current_pulsesequence_hash += pulse.serial

        # If they are the same, exit
        if self.current_pulsesequence_hash == self.last_pulsequence_hash:
            return

        sequence_start = 0
        sequence_end = 0
        
        # Find the start and end of the pulse sequence
        for channel_sequence in channel_pulses.values():
            for pulse in channel_sequence:
                sequence_start = min(pulse.start, sequence_start)
                sequence_end = max(pulse.start + pulse.duration, sequence_end)

        # Pad the start and end with zeros to meet the AWG minimum sample size count
        sequence_start = sequence_start * 1e-9 - self.awg.pulse_buffer
        sequence_end = sequence_end * 1e-9 + self.awg.pulse_buffer
        time_array = np.arange(sequence_start, sequence_end, 1 / self.awg.sample_rate)

        self.awg_waveform_buffer = np.zeros((4, len(time_array)))

        for fridge_port, channel_sequence in channel_pulses.items():
            # Get the IQ channels for the selected port
            i_ch, q_ch = self.channel_port_map[fridge_port]
            
            for pulse in channel_sequence:
                
                start = pulse.start * 1e-9
                duration = pulse.duration * 1e-9
                end = start + duration
                
                idx_start = bisect(time_array, start)
                idx_end = bisect(time_array, end)
                t = time_array[idx_start:idx_end]

                I = pulse.amplitude * np.cos(2 * np.pi * pulse.frequency * t + pulse.phase + self.awg.channel_phase[i_ch])
                Q = -pulse.amplitude * np.sin(2 * np.pi * pulse.frequency * t + pulse.phase + self.awg.channel_phase[q_ch])
                
                self.awg_waveform_buffer[i_ch, idx_start:idx_end] += I
                self.awg_waveform_buffer[q_ch, idx_start:idx_end] += Q

                # Store the readout pulse 
                if pulse.type == 'ro':
                    self.acquisitons.append((pulse.qubit, pulse.frequency))
                    self.readout_start = pulse.start

    def upload(self):
        """Uploads the pulse sequence to the AWG.
        """

        # Don't upload if the currently loaded pulses are the same as the previous set
        if self.current_pulsesequence_hash == self.last_pulsequence_hash:
            return

        if self.awg_waveform_buffer is None:
            raise_error(RuntimeError, "No pulse sequences currently configured")

        self.awg.upload(self.awg_waveform_buffer)
        self.last_pulsequence_hash = self.current_pulsesequence_hash

    def play_sequence(self):
        """Arms the AWG to play.
        """
        self.awg.play_sequence()

    def play_sequence_and_acquire(self):
        """Arms the AWG to play and the ADC to start acquisition.
        """

        self.play_sequence()
        self.adc.arm(self.nshots, self.readout_start)
        self.adc.acquire()
        results = {
            qubit_id: self.adc.process_result(readout_frequency)
            for qubit_id, readout_frequency in self.acquisitons
        }
        return results

    def stop(self):
        self.awg.stop()

    def disconnect(self):
        """Disconnects the AWG and ADC.
        """
        if self.is_connected:
            self.awg.disconnect()
            self.adc.disconnect()
            self.is_connected = False

    def start(self):
        pass
