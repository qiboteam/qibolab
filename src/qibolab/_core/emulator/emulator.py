import numpy as np
from qibo.config import log
from qutip import basis, destroy, mesolve, tensor

from qibolab import Readout
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

    def _play_sequence(self, sequence, configs, options, updates=None):
        """Play sequence on emulator."""

        if updates is None:
            updates = {}

        if any(isinstance(pulse, (Readout)) for _, pulse in sequence):
            print("Readout pulses parameter will be ignored except for duration.")

        hamiltonian = configs["hamiltonian"].hamiltonian
        hamiltonian += self.pulse_sequence_to_hamiltonian(sequence, configs, updates)
        measurement, tlist = self.measurement(sequence, configs, updates)

        results = mesolve(
            hamiltonian, self.initial_state, tlist, [], [self.probability]
        )
        averaged_results = {
            ro_pulse_id: results.expect[0][sample - 1]
            for ro_pulse_id, sample in measurement.items()
        }
        if options.averaging_mode == AveragingMode.SINGLESHOT:
            results = {
                ro_pulse_id: np.random.choice(
                    [0, 1], size=options.nshots, p=[1 - prob1, prob1]
                )
                for ro_pulse_id, prob1 in averaged_results.items()
            }
            return results
        return averaged_results

    def _sweep(self, sequence, configs, options, sweepers, updates=None):
        """Sweep over sequence."""
        results = {}
        sweeper = sweepers[0][0]
        if updates is None:
            updates = {}
        for value in sweeper.values:
            if sweeper.pulses is not None:
                for pulse in sweeper.pulses:
                    try:
                        updates[pulse.id].update({sweeper.parameter.name: value})
                    except KeyError:
                        updates[pulse.id] = {sweeper.parameter.name: value}
            if sweeper.channels is not None:
                for channel in sweeper.channels:
                    try:
                        updates[channel].update({sweeper.parameter.name: value})
                    except KeyError:
                        updates[channel] = {sweeper.parameter.name: value}
            if len(sweepers) > 1:
                temp = self._sweep(sequence, configs, options, sweepers[1:], updates)
            else:
                temp = self._play_sequence(sequence, configs, options, updates)

            results = merge_results(
                results,
                temp,
            )

        # reshaping results
        for key, value in results.items():
            results[key] = results[key].reshape(options.results_shape(sweepers))
        return results

    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list = None,
    ):
        assert len(sequences) == 1, "Emulator can only play one sequence at a time."
        assert (
            options.acquisition_type == AcquisitionType.DISCRIMINATION
        ), "Emulator only supports DISCRIMINATION acquisition type."

        if len(sweepers) > 0:
            sequence = sequences[0]
            results = self._sweep(sequence, configs, options, sweepers)
        else:
            results = self._play_sequence(sequences[0], configs, options)
        return results

    def measurement(self, sequence, configs, updates=None):
        """Given sequence creates a dictionary of readout pulses and their
        sample index."""
        duration = 0
        pulses = {}
        if updates is None:
            updates = {}

        for channel, pulse in sequence:
            if isinstance(configs[channel], AcquisitionConfig):
                if isinstance(pulse, Readout):
                    pulses[pulse.id] = int(duration)
                if pulse.id in updates:
                    pulse = pulse.model_copy(update=updates[pulse.id])
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
        """Construct Hamiltonian dependent term for qutip simulation."""

        hamiltonians = {}
        h_t = []

        for channel, pulse in sequence:
            # do not handle readout pulses
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


def waveform(pulse, channel, configs, updates=None):
    """Convert pulse to hamiltonian."""
    if updates is None:
        updates = {}
    # mapping IqConfig -> QubitDrive
    if isinstance(configs[channel], IqConfig):
        if channel in updates:
            config = configs[channel].model_copy(update=updates[channel])
            frequency = config.frequency
        else:
            frequency = configs[channel].frequency
        if pulse.id in updates:
            pulse = pulse.model_copy(update=updates[pulse.id])
        return QubitDrive(pulse=pulse, frequency=frequency)


def merge_results(a: dict, b: dict):
    """Merge results together."""
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    for key, value in b.items():
        a[key] = np.column_stack((a[key], value))
    return a
