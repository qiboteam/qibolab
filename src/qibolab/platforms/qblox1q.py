import copy
import itertools
import numpy as np
from qibolab import pulses, tomography
from qibolab.instruments import AcquisitionController
from qibolab.experiments.abstract import AbstractExperiment, ParameterList, BoundsValidator, EnumValidator, Qubit


# To be used for initial calibtation
qubit_static_parameters = [
    {
        "id": 0,
        "name": "Left/Bottom Qubit",
        "channel": [2, None, [0, 1]],  # XY control, Z line, readout
        "frequency_range": (3e9, 3.1e9),
        "resonator_frequency": 4.5172671e9,
        "amplitude": 0.375 / 2,
        "connected_qubits": [1]
    },
]

initial_calibration = [{
    "id": 0,
    "qubit_frequency": 3.06362669e9,
    "qubit_frequency_bounds": (3e9, 3.1e9),
    "resonator_frequency": 4.5172671e9,
    "resonator_frequency_bounds": (4.51e9, 4.525e9),
    "T1": 5.89e-6,
    "T2": 1.27e-6,
    "T2_Spinecho": 3.5e-6,
    "pi-pulse": 100.21e-9,
    "drive_channel": 2,
    "readout_channel": (0, 1),
    "connected_qubits": [1],
    "zero_iq_reference": (0.002117188393398148, 0.020081601323807922),
    "one_iq_reference": (0.007347951048047871, 0.015370747296983345),
    "initial_gates": {
        "rx": [pulses.BasicPulse(2, 0, 100.21e-9, 0.375 / 2, 3.06362669e9 - 2.3e9, 0, pulses.Rectangular()),
               pulses.BasicPulse(2, 0, 69.77e-9, 0.375 / 2, 3.086e9 - 2.3e9, 0, pulses.Rectangular())],
        "ry": [pulses.BasicPulse(2, 0, 100.21e-9, 0.375 / 2, 3.06362669e9 - 2.3e9, 90, pulses.Rectangular()),
               pulses.BasicPulse(2, 0, 69.77e-9, 0.375 / 2, 3.086e9 - 2.3e9, 90, pulses.Rectangular())],
        "measure": [pulses.BasicPulse(0, 0, 5e-6, 0.75 / 2, 100e6, 90, pulses.Rectangular()),  # I cosine
                    pulses.BasicPulse(1, 0, 5e-6, 0.75 / 2, 100e6, 0, pulses.Rectangular())],  # Q negative sine
        "cx_(1,)": [pulses.BasicPulse(3, 0, 46.71e-9, 0.396 / 2, 3.06362669e9 - 2.3e9, 0, pulses.SWIPHT(20e6))],
    }
}, ]


class qblox1q(AbstractExperiment):

    def __init__(self):
        super().__init__()
        self.name = "qblox1q"
        self.ac = AcquisitionController() # TODO: add QBLOX implementation
        self.ic = self.ac.ic
        self.results = None
        self.qubit_config = initial_calibration
        self.num_qubits = 1
        self.nchannels = 4

        self.readout_params = ParameterList()
        self.readout_params.add_parameter("LO_frequency",
                                          default=4.4172671e9,
                                          vals=(1e9, 10e9),
                                          validator=BoundsValidator)
        self.readout_params.add_parameter("attenuation",
                                          default=14,
                                          vals=(0, 35),
                                          validator=BoundsValidator)
        self.readout_params.add_parameter("ADC_delay",
                                          default=266e-9 + 16e-9)
        self.readout_params.add_parameter("RO_delay",
                                          default=266e-9 - 22e-9)
        self.readout_params.add_parameter("QB_delay",
                                          default=266e-9 + 26e-9)
        self.readout_params.add_parameter("ADC_sampling_rate",
                                          default=1e9)
        self.readout_params.add_parameter("ADC_sample_size",
                                          default=4992)
        self.readout_params.add_parameter("measurement_level",
                                          default=2,
                                          vals=[0, 1, 2],
                                          validator=EnumValidator)
        self.readout_params.add_parameter("duration",
                                          default=5e-6)
        self.readout_params.add_parameter("buffer",
                                          default=1e-6)
        self.readout_params.add_parameter("amplitude",
                                          default=0.75 / 2)
        self.readout_params.add_parameter("IF_frequency",
                                          default=100e6)

        self.awg_params = ParameterList()
        offset = [-0.001, 0, 0, 0]
        phase = [-6.2, 0.2, 0, 0]
        amplitude = [0.75, 0.75, 0.75, 0.75]
        self.awg_params.add_parameter("sampling_rate",
                                      default=2.3e9,
                                      vals=(1e9, 2.5e9),
                                      validator=BoundsValidator)
        for i in range(self.nchannels):
            self.awg_params.add_parameter("CH{}_offset".format(i + 1),
                                          default=offset[i])
            self.awg_params.add_parameter("CH{}_phase".format(i + 1),
                                          default=phase[i])
            self.awg_params.add_parameter("CH{}_amplitude".format(i + 1),
                                          default=amplitude[i])

        self.qubits = [Qubit(**qb) for qb in initial_calibration]

    def connect(self):
        pass

    def clock(self):
        pass

    def start(self, nshots):
        buffer, buffers_per_acquisition, records_per_buffer, samples_per_record, time_array = self.ac.do_acquisition()
        records_per_acquisition = (
            1. * buffers_per_acquisition * records_per_buffer)
        # Skip first 50 anomalous points
        recordA = np.zeros(samples_per_record - 50)
        recordB = np.zeros(samples_per_record - 50)

        for i in range(records_per_buffer):
            record_start = i * samples_per_record * 2
            record_stop = record_start + samples_per_record * 2
            record_slice = buffer[record_start:record_stop]
            recordA += record_slice[100::2] / records_per_acquisition
            recordB += record_slice[101::2] / records_per_acquisition

        recordA = self._signal_to_volt(recordA, 0.02)
        recordB = self._signal_to_volt(recordB, 0.02)

        self.results = [recordA, recordB]

    @staticmethod
    def _signal_to_volt(signal, voltdiv):
        u12 = signal / 16
        #bitsPerSample = 12
        codeZero = 2047.5
        codeRange = codeZero
        return voltdiv * (u12 - codeZero) / codeRange

    def stop(self):
        self.ac.stop()

    def _generate_readout_TTL(self, samples):
        end = self.readout_params.duration() + self.readout_params.buffer()
        duration = len(samples) / self.awg_params.sampling_rate()
        time_array = np.linspace(end - duration, end, num=samples)

        def TTL(t, start, duration, amplitude):
            x = amplitude * (1 * (start < t) & 1 * (start + duration > t))
            return x

        # ADC TTL
        start = self.readout_params.ADC_delay()
        adc_ttl = TTL(time_array, start, 10e-9, 1)

        # RO SW TTL
        start = self.readout_params.RO_delay()
        ro_ttl = TTL(time_array, start, self.readout_params.duration(), 1)

        # QB SW TTL
        start = self.readout_params.QB_SW_delay()
        qb_ttl = TTL(time_array, start, self.readout_params.duration(), 1)

        def square(t, start, duration, amplitude, freq, I_phase, Q_phase, *args, **kwargs):
            # Basic rectangular pulse
            x = amplitude * (1 * (start < t) & 1 * (start+duration > t))
            I_phase = I_phase * np.pi / 180
            Q_phase = Q_phase * np.pi / 180
            i = x * np.cos(2 * np.pi * freq * t + I_phase)
            q = - x * np.sin(2 * np.pi * freq * t + Q_phase)
            return i, q

        start = 0
        i_readout, q_readout = square(time_array, start, self.readout_params.duration(), self.readout_params.amplitude(),
                                      self.readout_params.IF_frequency(), -6.2, 0.2)

        return i_readout, q_readout, adc_ttl, ro_ttl, qb_ttl

    def upload(self, waveform, averaging):
        self.ic.setup(self.static.awg_params, self.static.lo_frequency,
                      self.static.qubit_attenuation, self.static.readout_attenuation, 0)
        self.ic.awg.set_nyquist_mode()
        ch3_drive = waveform[2]
        ch4_drive = waveform[3]
        #adc_ttl = waveform[4]
        #ro_ttl = waveform[5]
        #qb_ttl = waveform[6]
        i_readout, q_readout, adc_ttl, ro_ttl, qb_ttl = self._generate_readout_TTL(
            len(ch3_drive))
        #i_readout = waveform[0]
        #q_readout = waveform[1]
        output = self.ic.generate_pulse_sequence(
            i_readout, q_readout, ch3_drive, ch4_drive, adc_ttl, ro_ttl, qb_ttl, 20, averaging, self.awg_params.sampling_rate())
        self.ic.awg.upload_sequence(output, 1)
        self.ic.ready_instruments_for_scanning(
            7, self.readout_params.attenuation(), 0)
        self.ac.update_acquisitionkwargs(mode='NPT',
                                         samples_per_record=self.readout_params.ADC_length(),
                                         records_per_buffer=10,
                                         buffers_per_acquisition=int(
                                             averaging / 10),
                                         # channel_selection='AB',
                                         # transfer_offset=0,
                                         # external_startcapture='ENABLED',
                                         # enable_record_headers='DISABLED',
                                         # alloc_buffers='DISABLED',
                                         # fifo_only_streaming='DISABLED',
                                         interleave_samples='DISABLED',
                                         # get_processed_data='DISABLED',
                                         allocated_buffers=100,
                                         buffer_timeout=100000)

    def upload_batch(self, waveform_batch, averaging):
        self.ic.setup(self.static.awg_params, self.static.lo_frequency,
                      self.static.qubit_attenuation, self.static.readout_attenuation, 0)
        self.ic.awg.set_nyquist_mode()
        i_readout = waveform_batch[0, 0]
        q_readout = waveform_batch[1, 0]
        ch3_drive = waveform_batch[2]
        ch4_drive = waveform_batch[3]
        i_readout, q_readout, adc_ttl, ro_ttl, qb_ttl = self._generate_readout_TTL(
            len(i_readout))
        steps = len(ch3_drive)
        output = self.ic.generate_broadbean_sequence(
            i_readout, q_readout, ch3_drive, ch4_drive, steps, adc_ttl, ro_ttl, qb_ttl, 20, averaging, self.static.sampling_rate)
        self.ic.awg.upload_sequence(output, steps)
        self.ic.ready_instruments_for_scanning(
            7, self.readout_params.attenuation(), 0)
        self.ac.update_acquisitionkwargs(mode='NPT',
                                         samples_per_record=self.readout_params.ADC_sample_size(),
                                         records_per_buffer=10,
                                         buffers_per_acquisition=int(
                                             averaging / 10),
                                         # channel_selection='AB',
                                         # transfer_offset=0,
                                         # external_startcapture='ENABLED',
                                         # enable_record_headers='DISABLED',
                                         # alloc_buffers='DISABLED',
                                         # fifo_only_streaming='DISABLED',
                                         interleave_samples='DISABLED',
                                         # get_processed_data='DISABLED',
                                         allocated_buffers=100,
                                         buffer_timeout=100000)

    def start_batch(self, steps, nshots):
        self.results = np.zeros(
            (steps, 2, self.readout_params.ADC_sample_size() - 50))
        for k in range(steps):

            buffer, buffers_per_acquisition, records_per_buffer, samples_per_record, time_array = self.ac.do_acquisition()
            records_per_acquisition = (
                1. * buffers_per_acquisition * records_per_buffer)
            # Skip first 50 anomalous points
            recordA = np.zeros(samples_per_record - 50)
            recordB = np.zeros(samples_per_record - 50)

            for i in range(records_per_buffer):
                record_start = i * samples_per_record * 2
                record_stop = record_start + samples_per_record * 2
                record_slice = buffer[record_start:record_stop]
                recordA += record_slice[100::2] / records_per_acquisition
                recordB += record_slice[101::2] / records_per_acquisition

            self.results[k, 0] = self._signal_to_volt(recordA, 0.02)
            self.results[k, 1] = self._signal_to_volt(recordB, 0.02)

    def download(self):
        return self.results

    @staticmethod
    def check_tomography_required(target_qubits):
        return len(target_qubits) > 1

    def parse_raw(self, raw_signals, target_qubits):
        result = []
        final = self.readout_params.ADC_sample_size(
        ) / self.readout_params.ADC_sampling_rate()
        step = 1 / self.readout_params.ADC_sampling_rate()
        ADC_time_array = np.arange(0, final, step)[50:]

        for qubit in target_qubits:
            qubit_config = self.qubit_config[qubit]
            i_ch, q_ch = qubit_config["readout_channel"]
            i_sig = raw_signals[i_ch]
            q_sig = raw_signals[q_ch]
            IF_freq = qubit_config["readout_IF_frequency"]

            cos = np.cos(2 * np.pi * IF_freq * ADC_time_array)
            it = np.sum(i_sig * cos)
            qt = np.sum(q_sig * cos)
            result.append((it, qt))

        return result

    # Shallow method, to be reused for single shot measurement
    def parse_iq(self, iq_signals, target_qubits):
        prob = np.zeros(len(target_qubits))

        # First, we extract the probability p of finding each qubit in the 1 state
        # For each qubit, we assign p based on the distance between the zero and one state iq values and the measurement
        for idx, qubit in enumerate(target_qubits):
            data = np.array(iq_signals[idx])
            qubit_config = self.qubit_config[qubit]
            refer_0, refer_1 = qubit_config["iq_states"]

            move = copy.copy(refer_0)
            refer_0 = refer_0 - move
            refer_1 = refer_1 - move
            data = data - move
            # Rotate the data so that vector 0-1 is overlapping with Ox
            angle = copy.copy(np.arccos(
                refer_1[0]/np.sqrt(refer_1[0]**2 + refer_1[1]**2))*np.sign(refer_1[1]))
            new_data = np.array([data[0]*np.cos(angle) + data[1]*np.sin(angle),
                                -data[0]*np.sin(angle) + data[1]*np.cos(angle)])
            # Rotate refer_1 to get state 1 reference
            new_refer_1 = np.array([refer_1[0]*np.cos(angle) + refer_1[1]*np.sin(angle),
                                    -refer_1[0]*np.sin(angle) + refer_1[1]*np.cos(angle)])
            # Condition for data outside bound
            if new_data[0] < 0:
                new_data[0] = 0
            elif new_data[0] > new_refer_1[0]:
                new_data[0] = new_refer_1[0]
            prob[idx] = new_data[0] / new_refer_1[0]

        # Next, we process the probabilities into qubit states
        # Note: There are no correlations established here, this is solely for disconnected and unentangled qubits
        binary = list(itertools.product([0, 1], repeat=len(target_qubits)))
        result = np.zeros(len(binary)) + 1
        for idx, state in enumerate(binary):
            for idx2, qubit_state in state:
                p = prob[idx2]
                if qubit_state == 1:
                    p = 1 - p

                result[idx] *= p

        return result

    # Joint readout method
    def parse_tomography(self, iq_values, target_qubits):
        nqubits = len(target_qubits)
        nstates = int(2**nqubits)
        data = np.array([np.arctan2(q, i) * 180 / np.pi for i, q in iq_values])
        states = data[0:nstates]
        phase = data[nstates:len(iq_values)]
        tom = tomography.Tomography(phase, states)
        tom.minimize(1e-5)
        fit = tom.fit

        return fit
