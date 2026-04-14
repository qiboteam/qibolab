"""
In this module we often recall, for the sake of a better interpretation of the whole pipeline, the transformed dimensions of the simulation's results tensor.
Generally this tensor will have the shape (*S, M *H_dim) (or its permutations),
where:
- *S is the number of iteration for each sweep in the experiment
- M is the number of measurements applied in the pulse sequence
- *H_dim is the complete system dimension

In the results processing also, when simulating SINGLESHOTS, we'll add two dimensions:
- Nshots, which is simply the number of shots we average on
- M_unique, which is the number of unique measurement times
"""

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


def _extract_probabilities(states: NDArray) -> NDArray:
    """
    Calculate probabilities from a density matrix using diagonal elements.

    This function extracts the diagonal elements of each density matrix,
    which represent the probabilities of measurement outcomes.

    Probabilities are normalized for fluctuations, taking the absolute value of diagonal elements.

    Examples
    --------
    >>> dm = np.array([[0.9, 0.0], [0.0, 0.1]])
    >>> probs = _extract_probabilities(dm)
    >>> probs
    array([0.9, 0.1])
    """

    diag = np.einsum(
        states,
        # TODO: the `np.array()` wrapping call is only needed because of NumPy's type
        # annotation - in practice, it also works without
        np.array([...] + [0, 0]),
        np.array([...] + [0]),
    )
    return np.clip(diag.real, 0, 1)


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
) -> NDArray:
    """
    Select and organize quantum state acquisitions based on acquisition times.

    This function filters quantum states corresponding to specified acquisition times
    and maps them to unique acquisition values. It uses binary search to find the
    nearest state index for each acquisition time.

    It returns a NumPy array containing the full density matrices
    of the selected quantum states, indexed by the original acquisition order.
    """
    acq, index_pos = np.unique(list(acquisitions), return_inverse=True)
    samples = np.minimum(np.searchsorted(times, acq), times.size - 1)
    return np.stack([states[n].full() for n in samples])[index_pos]


def _cyclic_results(
    state_probs: NDArray,
    sequence: PulseSequence,
    hamiltonian: HamiltonianConfig,
    options: ExecutionParameters,
) -> dict[int, Result]:
    """Process measurement results from a cyclic quantum simulation, where the output for each measurement is
    excited state population.
    Computes readout results by projecting quantum state probabilities onto
    measurement subspaces and applying configured post-processing.
    """

    # Through the entire function state_probs has dimensions:
    # (*S, M *H_dim)
    states_computational_idx = np.stack(
        np.unravel_index(np.arange(state_probs.shape[-1]), hamiltonian.dims)
    )

    acq_id = acquisitions(sequence).keys()
    # from every acquisition pulse id we get the corresponding channel, and from the channel we get the
    # corresponding qubit index, which is then used to correctly permute the rows of states_computational_idx.
    qubit_indices = [
        index(sequence.pulse_channels(ro_id)[0], hamiltonian) for ro_id in acq_id
    ]
    permuted_states_computational_idx = states_computational_idx[qubit_indices]

    # applying a mask to select for each measurement the states that are outside the computational subspace, which are classified as 1
    mask = permuted_states_computational_idx >= 1

    # res is a (M, *S, ...) array
    res = np.moveaxis(np.sum(np.where(mask, state_probs, 0), axis=-1), -1, 0)

    if options.acquisition_type is AcquisitionType.INTEGRATION:
        res = np.random.normal(res, scale=0.001)
        zeros = np.zeros(res.shape) if np.ndim(res) != 0 else 0.0
        res = np.stack((res, zeros), axis=-1)

    return dict(zip(acq_id, res))


def _singleshot_results(
    state_probs: NDArray,
    sequence: PulseSequence,
    hamiltonian: HamiltonianConfig,
    options: ExecutionParameters,
) -> dict[int, Result]:
    """Extract measurement results from simulated quantum state probabilities.
    Performs single-shot measurement extraction from state probabilities by sampling
    according to the specified number of shots and mapping readout operations to their
    corresponding measurement results.
    """

    # select only unique times of measurements
    _, direct_map, inverse_map = np.unique(
        list(acquisitions(sequence).values()),
        return_index=True,
        return_inverse=True,
    )

    # here we move the -2 index of the probability array, hence it now becomes:
    # (M, *S, *H_dim)
    state_probs = np.moveaxis(state_probs, -2, 0)

    # apply the direct mapping found in np.unique to the probability vector
    # in order to sample at unique times and hence mantain correlation
    # since we use the same shots for synchronous measurements.
    unique_state_probs = state_probs[direct_map]

    # shots function returns a vector of shape:
    # (Nshots, M_unique, *S, *H_dim)
    sampled = shots(unique_state_probs, options.nshots)

    # move measurements dimension to the front, getting ready for extraction
    # the shape now is: (M_unique, Nshots, *S, *H_dim)
    sampled = np.moveaxis(sampled, 1, 0)

    acq_id = acquisitions(sequence).keys()
    # from every acquisition pulse id we get the corresponding channel, and from the channel we get the
    # corresponding qubit index, which is then used to correctly permute the rows of states_computational_idx.
    qubit_indices = [
        index(sequence.pulse_channels(ro_id)[0], hamiltonian) for ro_id in acq_id
    ]

    # we use inverse_map to expand back the sampled results
    # res is a (M, M, Nshots, *S, ...) array
    res = np.stack(np.unravel_index(sampled[inverse_map], hamiltonian.dims))[
        qubit_indices
    ]
    # using np.einsum, so res is a (M, Nshots, *S, ...) array
    res = np.einsum(res, np.array([0, 0] + [...]), np.array([0] + [...]))
    res = np.clip(res, 0, 1)

    if options.acquisition_type is AcquisitionType.INTEGRATION:
        zeros = np.zeros(res.shape) if np.ndim(res) != 0 else 0.0
        res = np.stack((res, zeros), axis=-1)

    return dict(zip(acq_id, res))


def results(
    states: NDArray,
    sequence: PulseSequence,
    hamiltonian: HamiltonianConfig,
    options: ExecutionParameters,
) -> dict[int, Result]:
    """Collect results for a single pulse sequence.

    The dictionary returned is already compliant with the expected
    result for the execution of this single sequence, thus suitable
    to be returned as is.
    """

    # probability dimensions are:
    # (*S, M, *H_dim)
    probabilities = _extract_probabilities(
        states,
    )

    results = (
        _singleshot_results
        if options.averaging_mode is AveragingMode.SINGLESHOT
        else _cyclic_results
    )

    return results(
        state_probs=probabilities,
        sequence=sequence,
        hamiltonian=hamiltonian,
        options=options,
    )
