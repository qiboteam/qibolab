import copy
import itertools
import numpy as np
from qibolab import pulses, tomography
from qibolab.pulse_shapes import Rectangular, SWIPHT
from qibolab.instruments.icarusqawg_controller import InstrumentController
from qibolab.experiments.abstract import AbstractExperiment, ParameterList, BoundsValidator, EnumValidator, Qubit


# AWG can only drive two qubits and read one line
qubit_static_parameters = [
    {
        "id": 0,
        "name": "Qubit 1",
        "channel": [2, None, [0, 1]], # XY control, Z line, readout
        "frequency_range": (6e9, 7e9),
        "resonator_frequency": 4.5e9,
        "amplitude": 0.375 / 2,
        "connected_qubits": [1]
    }, {
        "id": 1,
        "name": "Qubit 2",
        "channel": [3, None, [0, 1]],
        "frequency_range": (6e9, 7e9),
        "resonator_frequency": 5.5e9,
        "amplitude": 0.375 / 2,
        "connected_qubits": [0]
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
        "rx": [pulses.Pulse(0, 100.21e-9, 0.375 / 2, 3.06362669e9 - 2.3e9, 0, Rectangular(), channel=2),
                pulses.Pulse(0, 69.77e-9, 0.375 / 2, 3.086e9 - 2.3e9, 0, Rectangular(), channel=2)],
        "ry": [pulses.Pulse(0, 100.21e-9, 0.375 / 2, 3.06362669e9 - 2.3e9, 90, Rectangular(), channel=2),
                pulses.Pulse(0, 69.77e-9, 0.375 / 2, 3.086e9 - 2.3e9, 90, Rectangular(), channel=2)],
        "cx_(1,)": [pulses.Pulse(0, 46.71e-9, 0.396 / 2, 3.06362669e9 - 2.3e9, 0, SWIPHT(20e6), channel=3)],
        "measure": [pulses.IQReadoutPulse((0, 1), 0, 5e-6, 0.75 / 4, 100e6, (-6.2 / 180 * np.pi, 0.2 / 180 * np.pi))]
    }
}, {
    "id": 1,
    "qubit_frequency": 3.284049061e9,
    "qubit_frequency_bounds": (3.27e9, 3.29e9),
    "resonator_frequency": 4.5172671e9,
    "resonator_frequency_bounds": (4.51e9, 4.525e9),
    "T1": 5.89e-6,
    "T2": 1.27e-6,
    "T2_Spinecho": 3.5e-6,
    "pi-pulse": 112.16e-9,
    "drive_channel": 3,
    "readout_channel": (0, 1),
    "connected_qubits": [0],
    "zero_iq_reference": (0.002117188393398148, 0.020081601323807922),
    "one_iq_reference": (0.007347951048047871, 0.015370747296983345),
    "initial_gates": {
        "rx": [pulses.Pulse(0, 112.16e-9, 0.375 / 2, 3.284049061e9 - 2.3e9, 0, Rectangular(), channel=3),
                pulses.Pulse(0, 131.12e-9, 0.375 / 2, 3.23e9 - 2.3e9, 0, Rectangular(), channel=3)],
        "ry": [pulses.Pulse(0, 112.16e-9, 0.375 / 2, 3.284049061e9 - 2.3e9, 90, Rectangular(), channel=3),
                pulses.Pulse(0, 131.12e-9, 0.375 / 2, 3.23e9 - 2.3e9, 90, Rectangular(), channel=3)],
        "measure": [pulses.IQReadoutPulse((0, 1), 0, 5e-6, 0.75 / 4, 150e6, (-6.2 / 180 * np.pi, 0.2 / 180 * np.pi))]
    }
}]


class AWGSystem10Qubits(AbstractExperiment):

    def __init__(self):
        super().__init__()
        self.name = "awg10q"
        self.ic = InstrumentController()
        self.results = None
        self.qubit_config = initial_calibration
        self.num_qubits = 2
        self.nchannels = 4
        self.default_sample_size = None
        self.default_pulse_duration = 10e-6

        self.readout_params = ParameterList()
        self.readout_params.add_parameter("LO_frequency",
                                           default=4.35e9,
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
        self.readout_params.add_parameter("start_time",
                                          default=0)

        self.awg_params = ParameterList()
        offset = [-0.001, 0, 0, 0]
        phase = [-6.2, 0.2, 0, 0]
        amplitude = [0.75, 0.75, 0.75, 0.75]
        self.awg_params.add_parameter("sampling_rate",
                                      default=2.4e9,
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

        self.sampling_rate = self.awg_params.sampling_rate
        self.readout_pulse_duration = self.readout_params.duration

    def connect(self):
        pass

    def clock(self):
        pass

    def start(self, nshots):
        buffer, buffers_per_acquisition, records_per_buffer, samples_per_record = self.ic.do_acquisition()
        records_per_acquisition = (1. * buffers_per_acquisition * records_per_buffer)
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
        self.ic.stop()

    def _generate_TTL_pulses(self, samples):
        end = self.readout_params.duration() + self.readout_params.buffer()
        duration = samples / self.awg_params.sampling_rate()
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

        return adc_ttl, ro_ttl, qb_ttl

    def upload(self, waveform, averaging):
        self.ic.readout_attenuator.set_attenuation(0)
        self.ic.qubit_attenuator.set_attenuation(0)
        self.ic.lo.set_frequency(self.readout_params.LO_frequency())
        self.ic.awg.setup([self.awg_params["CH{}_offset".format(i + 1)] for i in self.nchannels], sampling_rate=self.awg_params.sampling_rate())

        sample_size = len(waveform[0])
        seq = np.zeros((1, self.nchannels, 3, sample_size))
        for i in range(self.nchannels):
            seq[0, i, 0] = waveform[i]
        adc_ttl, ro_ttl, qb_ttl = self._generate_TTL_pulses(sample_size)
        seq[0, 0, 1] = adc_ttl
        seq[0, 0, 2] = ro_ttl
        seq[0, 1, 2] = qb_ttl

        self.ic.awg.upload_sequence(seq, averaging)
        self.ic.update_adc(self.readout_params.ADC_sample_size(), averaging)

    def upload_batch(self, waveform_batch, averaging):
        self.ic.readout_attenuator.set_attenuation(self.readout_params.attenuation())
        self.ic.qubit_attenuator.set_attenuation(0)
        self.ic.lo.set_frequency(self.readout_params.LO_frequency())
        self.ic.awg.setup([self.awg_params["CH{}_offset".format(i + 1)] for i in self.nchannels], sampling_rate=self.awg_params.sampling_rate())

        sample_size = len(waveform_batch[0, 0])
        steps = len(waveform_batch)
        adc_ttl, ro_ttl, qb_ttl = self._generate_TTL_pulses(sample_size)
        seq = np.zeros((steps, self.nchannels, 3, sample_size))

        for idx, waveform in enumerate(waveform_batch):
            for i in range(self.nchannels):
                seq[idx][i, 0] = waveform[i]

            seq[idx, 0, 1] = adc_ttl
            seq[idx, 0, 2] = ro_ttl
            seq[idx, 1, 2] = qb_ttl

        self.ic.awg.upload_sequence(seq, averaging)
        self.ic.update_adc(self.readout_params.ADC_sample_size(), averaging)

    def start_batch(self, steps, nshots):
        self.results = np.zeros((steps, 2, self.readout_params.ADC_sample_size() - 50))
        for k in range(steps):

            buffer, buffers_per_acquisition, records_per_buffer, samples_per_record = self.ic.do_acquisition()
            records_per_acquisition = (1. * buffers_per_acquisition * records_per_buffer)
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
        final = self.readout_params.ADC_sample_size() / self.readout_params.ADC_sampling_rate()
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
            angle = copy.copy(np.arccos(refer_1[0]/np.sqrt(refer_1[0]**2 + refer_1[1]**2))*np.sign(refer_1[1]))
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
        res = self.parse_iq(iq_values, target_qubits)
        tom = tomography.Tomography(res)
        tom.minimize(1e-5)
        fit = tom.fit
        return fit
