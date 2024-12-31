import numpy as np
from qibo.config import log
from qutip import basis, destroy, mesolve, tensor

from qibolab import Parameter, Readout
from qibolab._core.components import AcquisitionConfig, Config, IqConfig
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.instruments.abstract import Controller
from qibolab._core.sequence import PulseSequence

from .model import QubitDrive


class EmulatorController(Controller):
    """Emulator controller."""

    bounds: str = "emulator/bounds"

    def connect(self):
        log.info("Starting emulator.")

    def disconnect(self):
        log.info("Stopping emulator.")

    @property
    def sampling_rate(self):
        """Sampling rate of emulator."""
        return 1

    @property
    def initial_state(self):
        """System in ground state."""
        # initial state: qubit in ground state
        return tensor(basis(2, 0))

    @property
    def probability(self):
        """Probability of qubit in excited state."""
        return destroy(2).dag() * destroy(2)

    def _play_sequence(self, sequence, configs, updates=None):
        """Play sequence on emulator."""

        if updates is None:
            updates = {}

        if any(isinstance(pulse, (Readout)) for _, pulse in sequence):
            print("Readout pulses parameter will be ignored except for duration.")

        hamiltonian = configs["hamiltonian"].hamiltonian
        hamiltonian += self.pulse_sequence_to_hamiltonian(sequence, configs, updates)
        measurement, tlist = self.measurement(sequence, configs)

        results = mesolve(
            hamiltonian, self.initial_state, tlist, [], [self.probability]
        )
        return {
            ro_pulse_id: results.expect[0][sample - 1]
            for ro_pulse_id, sample in measurement.items()
        }

    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list = None,
    ):
        assert len(sequences) == 1, "Emulator can only play one sequence at a time."
        assert len(sweepers) < 2, "Only up to 1 sweeper is supported."
        assert (
            options.averaging_mode == AveragingMode.CYCLIC
        ), "Emulator only supports CYCLIC averaging mode."
        assert (
            options.acquisition_type == AcquisitionType.DISCRIMINATION
        ), "Emulator only supports DISCRIMINATION acquisition type."

        if len(sweepers) == 1:
            sweeper = sweepers[0][0]
            assert sweeper.parameter in [
                Parameter.amplitude,
                Parameter.duration,
            ], "Emulator only supports amplitude or duration sweeps."
            sequence = sequences[0]
            results = {}
            for value in sweeper.values:
                updates = {}
                for pulse in sweeper.pulses:
                    updates[pulse.id] = {sweeper.parameter.name: value}
                temp = self._play_sequence(sequence, configs, updates)
                results = self.merge_results(results, temp)
        return results

    @staticmethod
    def merge_results(a: dict, b: dict):
        """Merge results together."""
        # TODO: check shape of results
        if len(a) == 0:
            return b
        if len(b) == 0:
            return a
        for key, value in b.items():
            if key in a:
                if isinstance(a[key], list):
                    a[key].append(value)
                else:
                    a[key] = [a[key], value]
        return a

    def measurement(self, sequence, configs):
        """Given sequence creates a dictionary of readout pulses and their
        sample index."""
        duration = 0
        pulses = {}
        for channel, pulse in sequence:
            if isinstance(configs[channel], AcquisitionConfig):
                if isinstance(pulse, Readout):
                    pulses[pulse.id] = int(duration)
                duration += pulse.duration

        tmax = int(max(pulses.values()) * self.sampling_rate)
        if tmax > 0:
            tlist = np.arange(0, tmax)
        else:
            tlist = np.arange(0, 1)
        return pulses, tlist

    def pulse_sequence_to_hamiltonian(
        self, sequence: PulseSequence, configs, updates=None
    ) -> dict[str, list]:
        """Construct Hamiltonian dependent term for quitip simulation."""

        hamiltonians = {}
        h_t = []

        def waveform(pulse, channel, configs, updates=None):
            """Convert pulse to hamiltonian."""
            if isinstance(configs[channel], IqConfig):
                frequency = configs[channel].frequency
                if updates is None:
                    updates = {}
                if pulse.id in updates:
                    pulse = pulse.model_copy(update=updates[pulse.id])
                return QubitDrive(pulse=pulse, frequency=frequency)

        for channel, pulse in sequence:
            if not isinstance(configs[channel], AcquisitionConfig):
                signal = waveform(pulse, channel, configs, updates)
                if channel in hamiltonians:
                    hamiltonians[channel] += [signal]
                else:
                    hamiltonians[channel] = [signal]

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
