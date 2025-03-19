"""Emulator controller."""

from collections import defaultdict
from collections.abc import Iterable
from functools import reduce
from operator import or_
from typing import Callable, Optional, cast

import numpy as np
from numpy.typing import NDArray
from qutip import Qobj, QobjEvo, mesolve

from qibolab._core.components import Config
from qibolab._core.components.configs import AcquisitionConfig
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.pulses import (
    Acquisition,
    Align,
    Delay,
    Pulse,
    PulseId,
    PulseLike,
    Readout,
    VirtualZ,
)
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers

from .hamiltonians import (
    HamiltonianConfig,
    Modulated,
    ModulatedVirtualZ,
    channel_operator,
    waveform,
)
from .utils import shots

__all__ = ["EmulatorController"]


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

    def _acquisitions(self, sequence: PulseSequence) -> dict[PulseId, float]:
        """Compute acqusitions' times."""
        acq = {}
        for ch in sequence.channels:
            time = 0
            for ev in sequence.channel(ch):
                if isinstance(ev, (Acquisition, Readout)):
                    acq[ev.id] = time
                if isinstance(ev, Align):
                    raise ValueError("Align not support in emulator.")
                time += ev.duration
        return acq

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

    def _play_sequence(
        self, sequence: PulseSequence, configs: dict[str, Config], updates: dict
    ) -> NDArray:
        """Play single sequence on emulator.

        The array returned by this function has a single dimension, over
        the various measurements included in the sequence.
        """
        sequence_ = update_sequence(sequence, updates)
        tlist_ = tlist(sequence_, self.sampling_rate)

        configs_ = update_configs(configs, updates)
        config = cast(HamiltonianConfig, configs_["hamiltonian"])
        hamiltonian = sum(config.hamiltonian)
        time_hamiltonian = self._pulse_hamiltonian(sequence_, configs_)
        if time_hamiltonian is not None:
            hamiltonian += time_hamiltonian

        results = mesolve(
            hamiltonian,
            config.initial_state,
            tlist_,
            config.dissipation,
            e_ops=[config.probability(state=i) for i in range(config.transmon_levels)],
        )

        return extract_probabilities(
            results.expect, self._acquisitions(sequence_).values(), tlist_
        )

    def _pulse_hamiltonian(
        self, sequence: PulseSequence, configs: dict[str, Config]
    ) -> Optional[QobjEvo]:
        """Construct Hamiltonian time dependent term for qutip simulation."""

        channels = [
            [operator, channel_time(waveforms)]
            for operator, waveforms in hamiltonians(sequence, configs)
        ]
        return QobjEvo(channels) if len(channels) > 0 else None


def update_sequence(sequence: PulseSequence, updates: dict) -> PulseSequence:
    """Apply sweep updates to base sequence."""
    return PulseSequence(
        [(ch, e.model_copy(update=updates.get(e.id, {}))) for ch, e in sequence]
    )


def update_configs(configs: dict[str, Config], updates: dict) -> dict[str, Config]:
    """Apply sweep updates to base configs."""
    return {k: c.model_copy(update=updates.get(k, {})) for k, c in configs.items()}


def tlist(
    sequence: PulseSequence, sampling_rate: float, per_sample: float = 2
) -> NDArray:
    """Compute times for evolution.

    The frequency of times is double the sampling rate, to make sure
    that all pulses features are resolved by the evolution.

    This can be customized using the `per_sample` rate, e.g. to retrieve times at the
    sampling rate itself, for pulses evaluation.

    .. note::

        As an optimization, if an acquisition is executed as the last
        sequence operation, that's not taken into account, since it is not
        simulated by the present emulator.

        For long experiments, it is a mild optimization. But it critically speeds up
        short experiments, given the usual relative duration of acquisitions and control
        pulses.
    """
    seq = (
        sequence[:-1]
        if isinstance(sequence[-1][1], (Acquisition, Readout))
        else sequence
    )
    end = max(seq.duration, 1)
    rate = sampling_rate * per_sample
    return np.arange(0, end, 1 / rate)


def hamiltonian(
    pulses: Iterable[PulseLike], config: Config, hamiltonian: HamiltonianConfig
) -> tuple[Qobj, list[Modulated]]:
    n = hamiltonian.transmon_levels
    op = channel_operator(n)
    waveforms = (
        waveform(pulse, config, n)
        for pulse in pulses
        # only handle pulses (thus no readout)
        if isinstance(pulse, (Pulse, Delay, VirtualZ))
    )
    return (op, [w for w in waveforms if w is not None])


def hamiltonians(
    sequence: PulseSequence, configs: dict[str, Config]
) -> Iterable[tuple[Qobj, list[Modulated]]]:
    hconfig = cast(HamiltonianConfig, configs["hamiltonian"])
    return (
        hamiltonian(sequence.channel(ch), configs[ch], hconfig)
        for ch in sequence.channels
        # TODO: drop the following, and treat acquisitions just as empty channels
        if not isinstance(configs[ch], AcquisitionConfig)
    )


def channel_time(waveforms: Iterable[Modulated]) -> Callable[[float], float]:
    """Wrap time function for specific channel.

    Used to avoid late binding issues.
    """

    def time(t: float) -> float:
        cumulative_time = 0
        cumulative_phase = 0
        for pulse in waveforms:
            pulse_duration = pulse.duration  # TODO: pass sampling rate
            pulse_phase = pulse.phase
            if cumulative_time <= t < cumulative_time + pulse_duration:
                relative_time = t - cumulative_time
                index = int(relative_time)  # TODO: pass sampling rate
                if isinstance(pulse, ModulatedVirtualZ):
                    continue
                return pulse(t, index, cumulative_phase)
            cumulative_time += pulse_duration
            # mirror rule used when compiling
            cumulative_phase -= pulse_phase
        return 0

    return time


def extract_probabilities(
    expectations: list[NDArray], acquisitions: Iterable[float], times: NDArray
) -> NDArray:
    """Extract probabilities from expectations.

    First, retrieve acquisitions, and locate them in the tlist, to
    isolate the expectations related to measurements.

    Then, it computes probabilities, based on the identified
    expectations.
    """
    acq = np.array(list(acquisitions))
    samples = np.minimum(np.searchsorted(times, acq), times.size - 1)
    return np.stack(expectations, axis=-1)[samples]
