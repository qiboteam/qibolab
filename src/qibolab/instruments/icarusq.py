import pyvisa as visa
import numpy as np
from typing import List, Optional, Union
from qcodes.instrument_drivers.AlazarTech import ATS

MODE_NYQUIST = 0
MODE_MIXER = 1

def square(t, start, duration, frequency, amplitude, phase):
    x = amplitude * (1 * (start < t) & 1 * (start+duration > t))
    i = x * np.cos(2 * np.pi * frequency * t + phase[0])
    q = - x * np.sin(2 * np.pi * frequency * t + phase[1])
    return i, q

def TTL(t, start, duration, amplitude):
    x = amplitude * (1 * (start < t) & 1 * (start + duration > t))
    return x

def sine(t, start, duration, frequency, amplitude, phase):
    x = amplitude * (1 * (start < t) & 1 * (start+duration > t))
    wfm = x * np.sin(2 * np.pi * frequency * t + phase)
    return wfm

class Instrument:
    def connect(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass

class VisaInstrument:
    def __init__(self) -> None:
        self._visa_handle = None

    def connect(self, address: str, timeout: int = 10000) -> None:
        rm = visa.ResourceManager()
        self._visa_handle = rm.open_resource(address, timeout=timeout)

    def write(self, msg: Union[bytes, str]) -> None:
        self._visa_handle.write(msg)

    def query(self, msg: Union[bytes, str]) -> str:
        return self._visa_handle.query(msg)

    def read(self) -> str:
        return self._visa_handle.read()

    def close(self) -> None:
        self._visa_handle.close()

    def ready(self) -> None:
        """
        Blocking command
        """
        self.query("*OPC?")

class TektronixAWG5204(VisaInstrument):

    def __init__(self, name, address):
        VisaInstrument.__init__(self)
        self.connect(address)
        self.name = name
        self._nchannels = 7
        self._sampling_rate = None
        self._mode = None
        self._amplitude = [0.75, 0.75, 0.75, 0.75]
        self._sequence_delay = None
        self._pulse_buffer = None
        self._adc_delay = None
        self._qb_delay = None
        self._ro_delay = None
        self._ip = None
        self._channel_phase = None

    def setup(self,
              offset: List[Union[int, float]],
              amplitude: Optional[List[Union[int, float]]] = [0.75, 0.75, 0.75, 0.75],
              resolution: Optional[int] = 14,
              sampling_rate: Optional[Union[int, float]] = 2.5e9,
              mode: int = MODE_MIXER,
              sequence_delay: float = 60e-6,
              pulse_buffer: float = 1e-6,
              adc_delay: float = 282e-9,
              qb_delay: float = 292e-9,
              ro_delay: float = 266e-9,
              ip: str = "192.168.0.2",
              channel_phase: List[float] = [-0.10821, 0.00349066, 0.1850049, -0.0383972],
              **kwargs) -> None:
        """ 
        Sets the channel offset, maximum amplitude, DAC resolution and sampling rate of the AWG
        """

        self.reset()
        for idx in range(4):
            ch = idx + 1
            self.write("SOURCe{}:VOLTage {}".format(ch, amplitude[idx]))
            self._amplitude[idx] = amplitude[idx]
            self.write("SOURCE{}:VOLTAGE:LEVEL:IMMEDIATE:OFFSET {}".format(ch, offset[ch - 1]))
            self.write("SOURce{}:DAC:RESolution {}".format(ch, resolution))

        self.write("SOUR1:DMOD NRZ")
        self.write("SOUR2:DMOD NRZ")
        self.write("CLOCk:SRATe {}".format(sampling_rate))

        if mode == MODE_NYQUIST:
            self.write("SOUR3:DMOD MIX")
            self.write("SOUR4:DMOD MIX")

        else:
            self.write("SOUR3:DMOD NRZ")
            self.write("SOUR4:DMOD NRZ")

        self._mode = mode
        self._sampling_rate = sampling_rate
        self._sequence_delay = 60
        self._pulse_buffer = pulse_buffer
        self._sequence_delay = sequence_delay
        self._qb_delay = qb_delay
        self._ro_delay = ro_delay
        self._adc_delay = adc_delay
        self._ip = ip
        self._channel_phase = channel_phase
        self.ready()

    def reset(self) -> None:
        self.write("INSTrument:MODE AWG")
        self.write("CLOC:SOUR EFIX") # Set AWG to external reference, 10 MHz
        self.write("CLOC:OUTP:STAT OFF") # Disable clock output
        self.clear()

    def clear(self) -> None:
        self.write('SLISt:SEQuence:DELete ALL')
        self.write('WLISt:WAVeform:DELete ALL')
        self.ready()

    def translate(self, sequence, shots):
        """
        Translates the pulse sequence into Tektronix .seqx file
        """

        import broadbean as bb
        from qibolab.pulses import ReadoutPulse
        from qcodes.instrument_drivers.tektronix.AWG70000A import AWG70000A

        # First create np arrays for each channel
        start = min(pulse.start for pulse in sequence)
        end = max(pulse.start + pulse.duration for pulse in sequence)
        t = np.arange(start * 1e-9 - self._pulse_buffer, end * 1e-9 + self._pulse_buffer, 1 / self._sampling_rate)
        wfm = np.zeros((self._nchannels, len(t)))

        for pulse in sequence:
            # Convert pulse timings from nanoseconds to seconds
            start = pulse.start * 1e-9
            duration = pulse.duration * 1e-9
            if isinstance(pulse, ReadoutPulse):
                # Readout IQ Signal
                i_ch = pulse.channel[0]
                q_ch = pulse.channel[1]
                phase = (self._channel_phase[i_ch] + pulse.phase, self._channel_phase[q_ch] + pulse.phase)
                i_wfm, q_wfm = square(t, start, duration, pulse.frequency, pulse.amplitude, phase)
                wfm[i_ch] += i_wfm
                wfm[q_ch] += q_wfm
                # ADC TTL
                wfm[4] = TTL(t, start + self._adc_delay , 10e-9, 1)
                # RO SW TTL
                wfm[5] = TTL(t, start + self._ro_delay, duration, 1)
                # QB SW TTL
                wfm[6] = TTL(t, start + self._qb_delay, duration, 1)

            else:
                if self._mode == MODE_MIXER:
                    # Qubit IQ signal
                    i_ch = pulse.channel[0]
                    q_ch = pulse.channel[1]
                    phase = (self._channel_phase[i_ch] + pulse.phase, self._channel_phase[q_ch] + pulse.phase)
                    i_wfm, q_wfm = square(t, start, duration, pulse.frequency, pulse.amplitude, phase)
                    wfm[i_ch] += i_wfm
                    wfm[q_ch] += q_wfm
                
                else:
                    qb_wfm = sine(t, start, duration, pulse.frequency, pulse.amplitude, pulse.phase)
                    wfm[pulse.channel] += qb_wfm

        # Add waveform arrays to broadbean sequencing
        main_sequence = bb.Sequence()
        main_sequence.name = "MainSeq"
        main_sequence.setSR(self._sampling_rate)

        dummy = np.zeros(len(t))
        unit_delay = 1e-6
        sample_delay = np.zeros(int(unit_delay * self._sampling_rate))
        delay_wfm = bb.Element()
        for ch in range(1, 5):
            delay_wfm.addArray(ch, sample_delay, self._sampling_rate, m1=sample_delay, m2=sample_delay)
        
        waveform = bb.Element()
        waveform.addArray(1, wfm[0], self._sampling_rate, m1=wfm[4], m2=wfm[5])
        waveform.addArray(2, wfm[1], self._sampling_rate, m1=dummy, m2=wfm[6])
        waveform.addArray(3, wfm[2], self._sampling_rate, m1=dummy, m2=dummy)
        waveform.addArray(4, wfm[3], self._sampling_rate, m1=dummy, m2=dummy)

        subseq = bb.Sequence()
        subseq.name = "SubSeq"
        subseq.setSR(self._sampling_rate)
        subseq.addElement(1, waveform)
        subseq.addElement(2, delay_wfm)
        subseq.setSequencingNumberOfRepetitions(2, int(self._sequence_delay / unit_delay))

        main_sequence.addSubSequence(1, subseq)
        main_sequence.setSequencingTriggerWait(1, 1)
        main_sequence.setSequencingNumberOfRepetitions(1, shots)
        main_sequence.setSequencingGoto(1, 1)

        payload = main_sequence.forge(apply_delays=False, apply_filters=False)
        payload = AWG70000A.make_SEQX_from_forged_sequence(payload, self._amplitude, "MainSeq")

        return payload

    def upload(self, payload):
        """
        Uploads the .seqx file to the AWG and loads it
        """
        import time
        with open("//{}/Users/OEM/Documents/MainSeq.seqx".format(self._ip), "wb+") as w:
            w.write(payload)

        pathstr = 'C:\\Users\\OEM\\Documents\\MainSeq.seqx'
        self.write('MMEMory:OPEN:SASSet:SEQuence "{}"'.format(pathstr))

        start = time.time()
        while True:
            elapsed = time.time() - start
            if int(self.query("*OPC?")) == 1:
                break
            elif elapsed > self._visa_handle.timeout:
                raise RuntimeError("AWG took too long to load waveforms")

        for ch in range(1, 5):
            self.write('SOURCE{}:CASSet:SEQuence "MainSeq", {}'.format(ch, ch))
        self.ready()

    def play_sequence(self):
        """
        Arms the AWG for playback on trigger A
        """
        for ch in range(1, 5):
            self.write("OUTPut{}:STATe 1".format(ch))
            self.write('SOURce{}:RMODe TRIGgered'.format(ch))
            self.write('SOURce1{}TINPut ATRIGGER'.format(ch))

        # Arm the trigger
        self.write('AWGControl:RUN:IMMediate')
        self.ready()

    def stop(self):
        """
        Stops the AWG and turns off all channels
        """
        self.write('AWGControl:STOP')
        for ch in range(1, 5):
            self.write("OUTPut{}:STATe 0".format(ch))

    def start_experiment(self):
        """
        Triggers the AWG to start playing
        """
        self.write('TRIGger:IMMediate ATRigger')        


class MCAttenuator(Instrument):

    def __init__(self, name, address):
        self.name = name
        self._address = address

    def setup(self, attenuation: int):
        import urllib3
        http = urllib3.PoolManager()
        http.request('GET', 'http://{}/SETATT={}'.format(self._address, attenuation))


class QuicSyn(VisaInstrument):

    def __init__(self, name, address):
        VisaInstrument.__init__(self)
        self.name = name
        self.connect(address)
        self.write('0601') # EXT REF

    def setup(self, frequency):
        """
        Sets the frequency in Hz
        """
        self.write('FREQ {0:f}Hz'.format(frequency))

    def start(self):
        self.write('0F01')

    def stop(self):
        self.write('0F00')

class AlazarADC(ATS.AcquisitionController, Instrument):
    def __init__(self, name="alz_cont", address="Alazar1", **kwargs):
        from qibolab.instruments.ATS9371 import AlazarTech_ATS9371
        
        self.adc = AlazarTech_ATS9371(address)
        self.acquisitionkwargs = {}
        self.samples_per_record = None
        self.records_per_buffer = None
        self.buffers_per_acquisition = None
        self.results = None
        self.number_of_channels = 2
        self.buffer = None
        self._samples = None
        self._thread = None
        self._processed_data = None
        super().__init__(name, address, **kwargs)
        self.add_parameter("acquisition", get_cmd=self.do_acquisition)


    def setup(self, samples):
        trigger_volts = 1
        input_range_volts = 2.5
        trigger_level_code = int(128 + 127 * trigger_volts / input_range_volts)
        with self.adc.syncing():
            self.adc.clock_source("EXTERNAL_CLOCK_10MHz_REF")
            #self.adc.clock_source("INTERNAL_CLOCK")
            self.adc.external_sample_rate(1_000_000_000)
            #self.adc.sample_rate(1_000_000_000)
            self.adc.clock_edge("CLOCK_EDGE_RISING")
            self.adc.decimation(1)
            self.adc.coupling1('DC')
            self.adc.coupling2('DC')
            self.adc.channel_range1(.02)
            #self.adc.channel_range2(.4)
            self.adc.channel_range2(.02)
            self.adc.impedance1(50)
            self.adc.impedance2(50)
            self.adc.bwlimit1("DISABLED")
            self.adc.bwlimit2("DISABLED")
            self.adc.trigger_operation('TRIG_ENGINE_OP_J')
            self.adc.trigger_engine1('TRIG_ENGINE_J')
            self.adc.trigger_source1('EXTERNAL')
            self.adc.trigger_slope1('TRIG_SLOPE_POSITIVE')
            self.adc.trigger_level1(trigger_level_code)
            self.adc.trigger_engine2('TRIG_ENGINE_K')
            self.adc.trigger_source2('DISABLE')
            self.adc.trigger_slope2('TRIG_SLOPE_POSITIVE')
            self.adc.trigger_level2(128)
            self.adc.external_trigger_coupling('DC')
            self.adc.external_trigger_range('ETR_2V5')
            self.adc.trigger_delay(0)
            #self.aux_io_mode('NONE') # AUX_IN_TRIGGER_ENABLE for seq mode on
            #self.aux_io_param('NONE') # TRIG_SLOPE_POSITIVE for seq mode on
            self.adc.timeout_ticks(0)

        self._samples = samples

            
    def update_acquisitionkwargs(self, **kwargs):
        """
        This method must be used to update the kwargs used for the acquisition
        with the alazar_driver.acquire
        :param kwargs:
        :return:
        """
        self.acquisitionkwargs.update(**kwargs)

    def arm(self, shots):
        import threading
        import time
        self.update_acquisitionkwargs(mode='NPT',
                                      samples_per_record=self._samples,
                                      records_per_buffer=10,
                                      buffers_per_acquisition=int(shots / 10),
                                      allocated_buffers=100,
                                      buffer_timeout=10000)
        self.pre_start_capture()
        self._thread = threading.Thread(target=self.do_acquisition, args=())
        self._thread.start()
        # TODO: Wait for armed flag instead of fixed time duration
        time.sleep(1)

    def pre_start_capture(self):
        self.samples_per_record = self.adc.samples_per_record.get()
        self.records_per_buffer = self.adc.records_per_buffer.get()
        self.buffers_per_acquisition = self.adc.buffers_per_acquisition.get()
        sample_speed = self.adc.get_sample_rate()
        t_final = self.samples_per_record / sample_speed
        self.time_array = np.arange(0, t_final, 1 / sample_speed)
        self.buffer = np.zeros(self.samples_per_record *
                               self.records_per_buffer *
                               self.number_of_channels)

    def pre_acquire(self):
        """
        See AcquisitionController
        :return:
        """
        # this could be used to start an Arbitrary Waveform Generator, etc...
        # using this method ensures that the contents are executed AFTER the
        # Alazar card starts listening for a trigger pulse
        pass


    def handle_buffer(self, data, buffer_number=None):
        """
        See AcquisitionController
        :return:
        """
        self.buffer += data

    def post_acquire(self):
        """
        See AcquisitionController
        :return:
        """

        def signal_to_volt(signal, voltdiv):
            u12 = signal / 16
            #bitsPerSample = 12
            codeZero = 2047.5
            codeRange = codeZero
            return voltdiv * (u12 - codeZero) / codeRange

        records_per_acquisition = (1. * self.buffers_per_acquisition * self.records_per_buffer)
        recordA = np.zeros(self.samples_per_record)
        recordB = np.zeros(self.samples_per_record)

        # Interleaved samples
        for i in range(self.records_per_buffer):
            record_start = i * self.samples_per_record * 2
            record_stop = record_start + self.samples_per_record * 2
            record_slice = self.buffer[record_start:record_stop]
            recordA += record_slice[0::2] / records_per_acquisition
            recordB += record_slice[1::2] / records_per_acquisition

        recordA = signal_to_volt(recordA, 0.02)
        recordB = signal_to_volt(recordB, 0.02)
        self._processed_data = np.array([recordA, recordB])
        return self.buffer, self.buffers_per_acquisition, self.records_per_buffer, self.samples_per_record, self.time_array

    def do_acquisition(self):
        """
        this method performs an acquisition, which is the get_cmd for the
        acquisiion parameter of this instrument
        :return:
        """
        self._get_alazar().acquire(acquisition_controller=self, **self.acquisitionkwargs)

    def result(self, readout_frequency):
        self._thread.join()

        # TODO: Pass ADC channel as arg instead of hardcoded channels
        input_vec_I = self._processed_data[0]
        input_vec_Q = self._processed_data[1]
        it = 0
        qt = 0
        for i in range(self.samples_per_record):
            it += input_vec_I[i] * np.cos(2 * np.pi * readout_frequency * self.time_array[i])
            qt += input_vec_Q[i] * np.cos(2 * np.pi * readout_frequency * self.time_array[i])
        phase = np.arctan2(qt, it) * 180 / np.pi
        ampl = np.sqrt(it**2 + qt**2)
        
        return ampl, phase, it, qt

    def close(self):
        self._alazar.close()
        super().close()
