from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from ...execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from ...identifier import ChannelId, Result
from ...pulses import Acquisition, PulseId, Readout
from ...sequence import PulseSequence
from .engine import Operator
from .hamiltonians import HamiltonianConfig


def ndchoice(probabilities: NDArray, samples: int) -> NDArray:
    """Sample elements with n-dimensional probabilities.

    This is the n-dimensional version of :func:`np.random.choice`, which instead of
    vectorizing over the picked elements, it assumes them to be just a plain integer
    range (which in turn could be used to index a suitable array, if relevant), while it
    allows the probabilities to be higher dimensional.

    The ``probabilities`` argument specifies the set of probabilities, which are
    intended to be normalized arrays over the innermost dimension. Such that the whole
    array describe a set of ``probabilities.shape[:-1]`` discrete distributions.

    `samples` is instead the number of samples to extract from each distribution.

    .. seealso::

        Generalized from https://stackoverflow.com/a/47722393, which presents the
        two-dimensional version.
    """
    np.random.seed(123456)
    return (
        probabilities.cumsum(-1).reshape(*probabilities.shape, -1)
        > np.random.rand(*probabilities.shape[:-1], 1, samples)
    ).argmax(-2)


def shots(probabilities: NDArray, nshots: int) -> NDArray:
    """Extract shots from state |0> ... |n> probabilities.

    This function just wraps :func:`ndchoice`, taking care of creating the n-D array of
    binomial distributions, and extracting shots as the outermost dimension.
    """
    shots = ndchoice(probabilities, nshots)
    # move shots from innermost to outermost dimension
    return np.moveaxis(shots, -1, 0)


def calculate_probabilities_from_density_matrix(states: NDArray) -> NDArray:
    """Compute probabilities from density matrix."""
    diag = np.einsum(
        states,
        # TODO: the `np.array()` wrapping call is only needed because of NumPy's type
        # annotation - in practice, it also works without
        np.array([...] + [0, 0]),
        np.array([...] + [0]),
    )
    return np.abs(diag)


def acquisitions(sequence: PulseSequence) -> dict[PulseId, float]:
    """Compute acquisitions' times."""
    acq = {}
    for ch in sequence.channels:
        time = 0
        for ev in sequence.channel(ch):
            if isinstance(ev, (Acquisition, Readout)):
                acq[ev.id] = time
            time += ev.duration

    return acq


def index(ch: ChannelId, hconfig: HamiltonianConfig) -> int:
    """Returns Hilbert space index from channel id."""
    if "coupler" in ch:
        target = int(ch.split("coupler_")[1].split("/")[0])
    else:
        target = int(ch.split("/")[0])
    return hconfig.hilbert_space_index(target)


def select_acquisitions(
    states: list[Operator], acquisitions: Iterable[float], times: NDArray
) -> tuple[NDArray, NDArray]:
    """Select density matrices from states.

    First, retrieve acquisitions, and locate them in the tlist, to
    isolate the expectations related to measurements.

    The return type should be rank-3 array, where the last two are the density
    matrices dimensions, while the first one should correspond to the acquisitions.
    """
    acq, index_pos = np.unique(list(acquisitions), return_inverse=True)
    samples = np.minimum(np.searchsorted(times, acq), times.size - 1)
    return np.stack([states[n].full() for n in samples]), index_pos


def results(
    states: NDArray,
    measurement_mapping: np.ndarray,
    sequence: PulseSequence,
    hamiltonian: HamiltonianConfig,
    options: ExecutionParameters,
) -> dict[int, Result]:
    """Collect results for a single pulse sequence.

    The dictionary returned is already compliant with the expected
    result for the execution of this single sequence, thus suitable
    to be returned as is.
    """
    probabilities = calculate_probabilities_from_density_matrix(
        states,
    )

    results = {}
    if options.averaging_mode is AveragingMode.CYCLIC:
        states_computational_idx = np.stack(
            np.unravel_index(np.arange(probabilities.shape[-1]), hamiltonian.dims)
        )

        for ro_dim, ro_id in zip(measurement_mapping, acquisitions(sequence).keys()):
            i = index(sequence.pulse_channels(ro_id)[0], hamiltonian)

            res = np.sum(
                probabilities[..., ro_dim, states_computational_idx[i] > 0], axis=-1
            )
            res = np.random.normal(res, scale=0.001)

            if options.acquisition_type is AcquisitionType.INTEGRATION:
                zeros = np.zeros(res.shape) if np.ndim(res) != 0 else 0.0
                res = np.stack((res, zeros), axis=-1)

            if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                res = np.clip(res, 0, 1)

            results[ro_id] = res

    if options.averaging_mode is AveragingMode.SINGLESHOT:
        assert options.nshots is not None
        sampled = shots(np.moveaxis(probabilities, -2, 0), options.nshots)
        # move measurements dimension to the front, getting ready for extraction
        measurements = np.moveaxis(sampled, 1, 0)
        # introduce cached measurements to avoid losing correlations
        cache_measurements = {}
        for (ro_id, sample_frame), ro_dim in zip(
            acquisitions(sequence).items(), measurement_mapping
        ):
            meas = measurements[ro_dim, :]
            i = index(sequence.pulse_channels(ro_id)[0], hamiltonian)
            cache_measurements.setdefault(sample_frame, meas)
            res = np.stack(
                np.unravel_index(cache_measurements[sample_frame], hamiltonian.dims)
            )[i]

            if options.acquisition_type is AcquisitionType.INTEGRATION:
                zeros = np.zeros(res.shape) if np.ndim(res) != 0 else 0.0
                res = np.stack((res, zeros), axis=-1)

            results[ro_id] = res

    return results
