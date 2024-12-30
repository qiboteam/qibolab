from qibolab._core.instruments.abstract import Controller
from qibolab._core.components import Config, IqConfig, AcquisitionConfig
from qibolab._core.sequence import PulseSequence
from qibolab._core.execution_parameters import ExecutionParameters, AveragingMode, AcquisitionType
from qibolab import Delay, Readout
from qutip import mesolve, tensor, basis, sigmaz, destroy
from qibo.config import log
from .model import QubitDrive
import numpy as np


class EmulatorController(Controller):
    """Emulator controller."""

    bounds: str = "emulator/bounds"

    def connect(self):
        log.info("Starting emulator.")

    def disconnect(self):
        log.info("Stopping emulator.")

    def setup(self, *args, **kwargs):
        pass

    @property
    def sampling_rate(self):
        """Sampling rate of emulator."""
        return 1

    @property
    def initial_state(self):
        """System in ground state."""
        # initial state: qubit in ground state
        return tensor(basis(2,0))

    @property
    def probability(self):
        """Probability of qubit in excited state."""
        return destroy(2).dag()*destroy(2)

    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list = None,
    ):
        assert len(sequences) == 1, "Emulator can only play one sequence at a time."
        assert len(sweepers) == 0, "Emulator does not support sweepers."
        assert options.averaging_mode == AveragingMode.CYCLIC, "Emulator only supports CYCLIC averaging mode."
        assert options.acquisition_type == AcquisitionType.DISCRIMINATION, "Emulator only supports DISCRIMINATION acquisition type."

        sequence = sequences[0]
        if any(isinstance(pulse, (Readout)) for _, pulse in sequence):
            print("Readout pulses parameter will be ignored.")

        hamiltonian = []

        # time independent hamiltonian
        for config in configs.values():
            if config.kind == "qubit":
                hamiltonian.append(config.operator)

        hamiltonian += self.pulse_sequence_to_hamiltonian(sequence, configs)
        measurement = self.measurement(sequence, configs)

        # quick fix for 0 tmax
        tmax = int(max(measurement.values())*self.sampling_rate)
        if tmax > 0:
            tlist = np.arange(0, tmax)
        else:
            tlist = np.arange(0, 1)

        results = mesolve(hamiltonian, self.initial_state, tlist, [], [self.probability])
        return {ro_pulse_id: results.expect[0][sample-1] for ro_pulse_id, sample in measurement.items()}

    @staticmethod
    def measurement(sequence, configs):
        """Given sequence creates a dictionary of readout pulses and their sample index."""
        duration = 0
        pulses = {}
        for channel, pulse in sequence:
            if isinstance(configs[channel], AcquisitionConfig):
                if isinstance(pulse, Readout):
                    pulses[pulse.id] = int(duration)
                duration += pulse.duration
        return pulses


    def pulse_sequence_to_hamiltonian(self, sequence: PulseSequence, configs) -> dict[str, list]:
        """Construct Hamiltonian dependent term for quitip simulation."""

        hamiltonians = {}
        h_t = []

        def waveform(pulse, channel, configs):
            """Convert pulse to hamiltonian"""
            if isinstance(configs[channel], IqConfig):
                frequency = configs[channel].frequency
                return QubitDrive(pulse=pulse, frequency=frequency)

        for channel, pulse in sequence:
            if not isinstance(configs[channel], AcquisitionConfig):
                if channel in hamiltonians:
                    hamiltonians[channel] += [waveform(pulse, channel, configs)]
                else:
                    hamiltonians[channel] = [waveform(pulse, channel, configs)]

        for channel, waveforms in hamiltonians.items():
            def time(t, args=None):
                result = find_sample_array(t, *waveforms)
                if result is None:
                    return 0
                waveform, sample = result
                return waveform(t, sample)

            h_t.append([waveforms[0].operator, time])
        return h_t


def find_sample_array(sample_index, *arrays):
    """Basic function to recover array and sample index."""
    current_index = 0
    for i, array in enumerate(arrays):
        if current_index <= sample_index < current_index + len(array):
            return arrays[i], int(sample_index - current_index)
        current_index += len(array)
    return None