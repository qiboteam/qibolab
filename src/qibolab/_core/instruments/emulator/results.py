from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from ...execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from ...identifier import ChannelId, QubitId, Result
from ...pulses import Acquisition, Align, PulseId, Readout
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


def calculate_probabilities_from_density_matrix(
    states: NDArray, subsystems: Iterable[QubitId], nsubsystems: int, d: int
) -> NDArray:
    """Compute probabilities from density matrix."""
    states_ = np.reshape(states, states.shape[:-2] + 2 * nsubsystems * (d,))
    marginal = np.einsum(
        states_,
        # TODO: the `np.array()` wrapping call is only needed because of NumPy's type
        # annotation - in practice, it also works without
        np.array([...] + list(range(nsubsystems)) * 2),
        np.array([...] + list(subsystems)),
    )
    return np.abs(marginal).reshape((*states.shape[:-2], -1))


def acquisitions(sequence: PulseSequence) -> dict[PulseId, float]:
    """Compute acquisitions' times."""
    acq = {}
    for ch in sequence.channels:
        time = 0
        for ev in sequence.channel(ch):
            if isinstance(ev, (Acquisition, Readout)):
                acq[ev.id] = time
            assert not isinstance(ev, Align)
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

    It returns a tuple containing 1 NumPy array containing the full density matrices
    of the selected quantum states.
    """
    acq = np.array(list(acquisitions))
    samples = np.minimum(np.searchsorted(times, acq), times.size - 1)
    return np.stack([states[n].full() for n in samples])


def cyclic_results(
    state_probs: NDArray,
    sequence: PulseSequence,
    hamiltonian: HamiltonianConfig,
    options: ExecutionParameters,
) -> dict[int, Result]:
    """Process measurement results from a cyclic quantum simulation, where the output for each measurement is
    excited state population.
    Computes readout results by projecting quantum state probabilities onto
    measurement subspaces and applying configured post-processing.

    Notes:
        - States outside the computational subspace (values > 1) are classified as 1.
        - For integration acquisition type, imaginary components are set to zero.
    """

    # Through the entire function state_probs has dimensions:
    # (M, S_i, H_dim), where
    # M is the number of measurements applied in the pulse sequence
    # S_i is the number of iteration for each sweep in the experiment
    # H_dim is the complete system dimension
    states_computational_idx = np.stack(
        np.unravel_index(np.arange(state_probs.shape[-1]), hamiltonian.dims)
    )

    res_dict = {}
    for meas_ro, ro_id in zip(state_probs, acquisitions(sequence).keys()):
        i = index(sequence.pulse_channels(ro_id)[0], hamiltonian)

        res = np.sum(meas_ro[..., states_computational_idx[i] > 0], axis=-1)
        res = np.random.normal(res, scale=0.001)

        if options.acquisition_type is AcquisitionType.INTEGRATION:
            zeros = np.zeros(res.shape) if np.ndim(res) != 0 else 0.0
            res = np.stack((res, zeros), axis=-1)

        if options.acquisition_type is AcquisitionType.DISCRIMINATION:
            res = np.clip(res, 0, 1)

        # res is a (S_i, ...) array
        res_dict[ro_id] = res

    return res_dict


def singleshot_results(
    state_probs: NDArray,
    sequence: PulseSequence,
    hamiltonian: HamiltonianConfig,
    options: ExecutionParameters,
) -> dict[int, Result]:
    """Extract measurement results from simulated quantum state probabilities.
    Performs single-shot measurement extraction from state probabilities by sampling
    according to the specified number of shots and mapping readout operations to their
    corresponding measurement results.

    Notes:
        - States outside the computational subspace (values > 1) are classified as 1.
        - For integration acquisition type, imaginary components are set to zero.
    """

    # select only unique times of measurements
    _, direct_map, inverse_map = np.unique(
        list(acquisitions(sequence).values()),
        return_index=True,
        return_inverse=True,
    )

    # apply the direct mapping found in np.unique to the probability vector
    # in order to sample at unique times and hence mantain correlation
    # since we use the same shots for synchronous measurements.
    unique_state_probs = state_probs[direct_map]

    # shots function returns a vector of shape:
    # (Nshots, M, S_i, H_dim), where
    # Nshots is simply the number of shots we average on
    # M is the number of measurements applied in the pulse sequence
    # S_i is the number of iteration for each sweep in the experiment
    # H_dim is the complete system dimension
    sampled = shots(unique_state_probs, options.nshots)

    # move measurements dimension to the front, getting ready for extraction
    # the shape now is: (M, Nshots, S_i, H_dim)
    sampled = np.moveaxis(sampled, 1, 0)

    res_dict = {}
    for ro_id, inv_idx in zip(acquisitions(sequence).keys(), inverse_map):
        i = index(sequence.pulse_channels(ro_id)[0], hamiltonian)
        # states out of the qubit computational space are classified as 1
        # here sampled has dimensions (M, Nshots, S_i, H_dim)
        res = np.clip(np.unravel_index(sampled[inv_idx], hamiltonian.dims)[i], 0, 1)

        if options.acquisition_type is AcquisitionType.INTEGRATION:
            zeros = np.zeros(res.shape) if np.ndim(res) != 0 else 0.0
            res = np.stack((res, zeros), axis=-1)

        # res is a (Nshots, S_i, ...) array
        res_dict[ro_id] = res

    return res_dict


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
    # (S_i, M, H_dim), where
    # S_i is the number of iteration for each sweep in the experiment
    # M is the number of measurements applied in the pulse sequence
    # H_dim is the complete system dimension
    probabilities = calculate_probabilities_from_density_matrix(
        states,
        range(hamiltonian.nqubits),
        hamiltonian.nqubits,
        hamiltonian.transmon_levels,
    )
    assert options.nshots is not None
    sampled = shots(np.moveaxis(probabilities, -2, 0), options.nshots)
    # move measurements dimension to the front, getting ready for extraction
    measurements = np.moveaxis(sampled, 1, 0)

    # here we move the -2 index of the probability array, hence it now becomes:
    # (M, S_i, H_dim)
    probabilities = np.moveaxis(probabilities, -2, 0)

    results = (
        singleshot_results
        if options.averaging_mode is AveragingMode.SINGLESHOT
        else cyclic_results
    )
    assert (options.averaging_mode is not AveragingMode.SINGLESHOT) or (
        options.nshots is not None
    ), "nshots must be specified for SINGLESHOT mode"
    return results(
        state_probs=probabilities,
        sequence=sequence,
        hamiltonian=hamiltonian,
        options=options,
    )
