import copy
import itertools
import numpy as np
from io import BytesIO
from qibo.config import log, raise_error
from qiboicarusq import connections, pulses
from qiboicarusq.experiments.abstract import AbstractExperiment


class IcarusQ(AbstractExperiment):

    class StaticParameters():
        """Hardware static parameters."""
        num_qubits = 2
        sampling_rate = 2.3e9
        nchannels = 4
        sample_size = 32000
        readout_pulse_type = "IQ"
        readout_pulse_duration = 5e-6
        readout_pulse_amplitude = 0.75
        lo_frequency = 4.51e9
        readout_nyquist_zone = 4
        ADC_sampling_rate = 2e9
        default_averaging = 10000
        qubit_static_parameters = [
            {
                "id": 0,
                "channel": [2, None, [0, 1]], # XY control, Z line, readout
                "frequency_range": [2.6e9, 2.61e9],
                "resonator_frequency": 4.5241e9,
                "neighbours": [2],
                "amplitude": 0.75 / 2
            }, {
                "id": 1,
                "channel": [3, None, [0, 1]],
                "frequency_range": [3.14e9, 3.15e9],
                "resonator_frequency": 4.5241e9,
                "neighbours": [1],
                "amplitude": 0.75 / 2
            }
        ]
        dac_mode_for_nyquist = ["NRZ", "MIX", "MIX", "NRZ"] # fifth onwards not calibrated yet
        pulse_file = 'C:/fpga_python/fpga/tmp/wave_ch1.csv'

        # Initial calibrated parameters to speed up calibration  
        initial_calibration = [{
            "id": 0,
            "qubit_frequency": 3.0473825e9,
            "qubit_amplitude": 0.75 / 2,
            "T1": 5.89e-6,
            "T2": 1.27e-6,
            "T2_Spinecho": 3.5e-6,
            "pi-pulse": 24.78e-9,
            "drive_channel": 3,
            "readout_channel": (0, 1),
            "iq_state": {
                "0": [0.016901687416102748, -0.006633150376482062],
                "1": [0.009458352995780546, -0.008570922209494462]
            },
            "gates": {
                "rx": [pulses.BasicPulse(3, 0, 24.78e-9, 0.375, 3.0473825e9 - sampling_rate, 0, pulses.Rectangular())],
                "ry": [pulses.BasicPulse(3, 0, 24.78e-9, 0.375, 3.0473825e9 - sampling_rate, 90, pulses.Rectangular())],
            }
        }]

        # Same standard as OpenPulses measurement output level
        # Level 0: Raw Signal, IQ Values and Qubit State
        # Level 1: IQ Values and Qubit State
        # Level 2: Qubit State
        measurement_level = 2

    def __init__(self):
        super().__init__()
        self.name = "icarusq"
        self.static = self.StaticParameters()

    def connect(self, address, username, password):
        self._connection = connections.ParamikoSSH(address, username, password)

    def clock(self):
        self.connection.exec_command('clk-control')

    def start(self, nshots, adc_delay=0.0, verbose=False):
        stdin, stdout, stderr = self.connection.exec_command( # pylint: disable=E1111
            'cd /tmp; ./cqtaws 1 {:.06f}'.format(adc_delay * 1e6))  # delay in us
        if verbose:
            for line in stdout:
                log.info(line.strip('\n'))

    def stop(self):
        self.connection.exec_command('cd /tmp; ./cqtaws 0 0')

    def upload(self, waveform):
        dump = BytesIO()
        with self.connection as sftp:
            for i in range(self.static.nchannels):
                dump.seek(0)
                np.savetxt(dump, waveform[i], fmt='%d', newline=',')
                dump.seek(0)
                sftp.putfo(dump)
        dump.close()

    def download(self):
        waveform = np.zeros((self.static.nchannels, self.static.sample_size))
        dump = BytesIO()
        with self.connection as sftp:
            for i in range(self.static.nchannels):
                dump.seek(0)
                #sftp.get('/tmp/ADC_CH{}.txt'.format(i + 1), local + 'ADC_CH{}.txt'.format(i + 1))
                dump = sftp.getfo(dump)
                dump.seek(0)
                #waveform.append(np.genfromtxt(local + 'ADC_CH{}.txt', delimiter=',')[:-1])
                waveform[i] = np.genfromtxt(dump, delimiter=',')[:-1]
        dump.close()

        return waveform

    @staticmethod
    def check_tomography_required(target_qubits):
        # TODO: Implement check for connectivity on the 10 qubit chip
        return len(target_qubits) > 1

    def parse_raw(self, raw_signals, target_qubits):
        result = []
        final = self.static.ADC_length / self.static.ADC_sampling_rate
        step = 1 / self.static.ADC_sampling_rate
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
        raise_error(NotImplementedError)
