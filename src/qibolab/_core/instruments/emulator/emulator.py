"""Emulator controller."""

from collections import defaultdict
from functools import reduce
from operator import or_
from typing import Optional, cast

import numpy as np
from numpy.typing import NDArray
from qutip import mesolve

from qibolab._core.components import AcquisitionConfig, Config
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.pulses import Acquisition, Align, PulseId, Readout
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers

from .hamiltonians import HamiltonianConfig, waveform
from .utils import shots

__all__ = ["EmulatorController"]


def measurements(sequence: PulseSequence) -> dict[PulseId, float]:
    """Extract acquisition identifiers and durations."""
    return {acq.id: acq.duration for _, acq in sequence.acquisitions}


def update_sequence(sequence: PulseSequence, updates: dict) -> PulseSequence:
    """Apply sweep updates to base sequence."""
    return PulseSequence(
        [(ch, e.model_copy(update=updates.get(e.id, {}))) for ch, e in sequence]
    )


class EmulatorController(Controller):
    """Emulator controller."""

    bounds: str = "emulator/bounds"

    def connect(self):
        """Dummy connect method."""

    def disconnect(self):
        """Dummy disconnect method."""

    @property
    def sampling_rate(self):
        """Sampling rate of emulator."""
        return 1

    def _play_sequence(
        self, sequence: PulseSequence, configs: dict[str, Config], updates: dict
    ) -> NDArray:
        """Play single sequence on emulator.

        The array returned by this function has a single dimension, over
        the various measurements included in the sequence.
        """
        config = cast(HamiltonianConfig, configs["hamiltonian"])
        sequence_ = update_sequence(sequence, updates)
        hamiltonian = config.hamiltonian
        hamiltonian += self._pulse_sequence_to_hamiltonian(sequence_, configs, updates)
        tlist = np.arange(
            0, max(measurements(sequence_).values()), 1 / self.sampling_rate / 2
        )
        results = mesolve(
            hamiltonian,
            config.initial_state,
            tlist,
            config.dissipation,
            e_ops=[config.probability(state=i) for i in range(config.transmon_levels)],
        )
        acq = np.array(list(self._acquisitions(sequence_).values()))
        return np.stack(
            [results.expect[i][acq] for i in range(config.transmon_levels)], axis=-1
        )

    def _sweep(
        self,
        sequence: PulseSequence,
        configs: dict[str, Config],
        sweepers: list[ParallelSweepers],
        updates: Optional[dict] = None,
    ) -> NDArray:
        """Sweep over sequence.

        This function invokes itself recursively, adding an array
        dimension at each call as the outermost one. The extra dimension
        corresponds to the values in the first nested sweep (with the
        lowest index, interpreted as the outermost as well).
        """
        # use a default dictionary, merging existing values
        updates = defaultdict(dict) | ({} if updates is None else updates)

        if len(sweepers) == 0:
            return self._play_sequence(sequence, configs, updates)

        parsweep = sweepers[0]
        # collect slices of results, corresponding to the current iteration
        results = []
        # execute once for each parallel value
        for values in zip(*(s.values for s in parsweep)):
            # update all parallel sweepers with the respective values
            for sweeper, value in zip(parsweep, values):
                if sweeper.pulses is not None:
                    for pulse in sweeper.pulses:
                        updates[pulse.id].update({sweeper.parameter.name: value})
                if sweeper.channels is not None:
                    for channel in sweeper.channels:
                        updates[channel].update({sweeper.parameter.name: value})

            # append new slice for the current parallel value
            results.append(self._sweep(sequence, configs, sweepers[1:], updates))

        # stack all slices in a single array, along the current outermost dimension
        return np.stack(results)

    def _single_sequence(
        self,
        sequence: PulseSequence,
        configs: dict[str, Config],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[int, Result]:
        """Collect results for a single pulse sequence.

        The dictionary returned is already compliant with the expected
        result for the execution of this single sequence, thus suitable
        to be returned as is.
        """
        # probabilities for the |0>, |1> ... |n> state, for each swept value
        probabilities = self._sweep(sequence, configs, sweepers)
        assert options.nshots is not None
        # extract results from probabilities, according to the requested averaging mode
        averaged = (
            shots(probabilities, options.nshots)
            if options.averaging_mode == AveragingMode.SINGLESHOT
            # weighted averaged
            else np.sum(probabilities * np.arange(probabilities.shape[-1]), axis=-1)
        )
        # move measurements dimension to the front, getting ready for extraction
        res = np.moveaxis(averaged, -1, 0)

        if options.acquisition_type is AcquisitionType.DISCRIMINATION:
            measurements = res
        elif options.acquisition_type is AcquisitionType.INTEGRATION:
            x = np.stack((res, np.zeros_like(res)), axis=-1)
            measurements = np.random.normal(x, scale=0.001)
        else:
            raise ValueError(
                f"Acquisition type '{options.acquisition_type}' unsupported"
            )
        # match measurements with their IDs, in order to already comply with the general
        # format established by the `Controller` interface
        measurement_ids = self._acquisitions(sequence).keys()
        return dict(zip(measurement_ids, list(measurements)))

    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[int, Result]:
        # just merge the results of multiple executions in a single dictionary
        return reduce(
            or_,
            (
                self._single_sequence(sequence, configs, options, sweepers)
                for sequence in sequences
            ),
        )

    def _acquisitions(self, sequence: PulseSequence) -> dict[PulseId, float]:
        """Compute measurements' times."""
        meas = {}
        for ch in sequence.channels:
            duration = 0
            for ev in sequence.channel(ch):
                if isinstance(ev, (Acquisition, Readout)):
                    meas[ev.id] = duration
                if isinstance(ev, Align):
                    raise ValueError("Align not support in emulator.")
                duration += ev.duration
        return meas

    def _pulse_sequence_to_hamiltonian(
        self, sequence: PulseSequence, configs: dict[str, Config], updates: dict
    ) -> dict[str, list]:
        """Construct Hamiltonian dependent term for qutip simulation."""

        hamiltonians = defaultdict(list)
        h_t = []
        for channel, pulse in sequence:
            # do not handle readout pulses
            if not isinstance(configs[channel], AcquisitionConfig):
                signal = waveform(pulse, channel, configs, updates)
                hamiltonians[channel] += [signal]

        for channel, waveforms in hamiltonians.items():

            def _wrapped_time(waveforms):
                """Wrapped time function for specific channel.

                Used to avoid late binding issues.
                """

                def time(t):
                    cumulative_time = 0
                    for pulse in waveforms:
                        pulse_duration = len(pulse)  # TODO: pass sampling rate
                        if cumulative_time <= t < cumulative_time + pulse_duration:
                            relative_time = t - cumulative_time
                            index = int(relative_time)  # TODO: pass sampling rate
                            return pulse(t, index)
                        cumulative_time += pulse_duration
                    return 0

                return time

            h_t.append([waveforms[0].operator, _wrapped_time(waveforms)])
        return h_t
