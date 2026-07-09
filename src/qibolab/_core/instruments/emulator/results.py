"""
In this module we often recall, for the sake of a better interpretation of the whole pipeline, the transformed dimensions of the simulation's results tensor.
Generally this tensor will have the shape (*S, M *H_dim) (or its permutations),
where:
- *S is the number of iteration for each sweep in the experiment
- M is the number of measurements applied in the pulse sequence
- *H_dim is the complete system dimension
- H_i is the dimension of the i-th subspace (associated to the i-th qubit)

In the results processing also, when simulating SINGLESHOTS, we'll add two dimensions:
- Nshots, which is simply the number of shots we average on
- M_unique, which is the number of unique measurement times
"""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import ChannelId, Result
from qibolab._core.pulses import Acquisition, Align, PulseId, Readout
from qibolab._core.sequence import PulseSequence

from .engine import Operator
from .hamiltonians import HamiltonianConfig, Qubit, QubitId


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

    Probabilities are normalized for fluctuations, taking the real value of diagonal elements.

    Examples
    --------
    >>> dm = np.array([[0.9, 0.0], [0.0, 0.1]])
    >>> probs = _extract_probabilities(dm)
    >>> probs
    array([0.9, 0.1])
    """

    diag = np.einsum(
        states,
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
            assert not isinstance(ev, Align)
            time += ev.duration

    return acq


def qubit_id(ch: ChannelId) -> QubitId:
    """Returns qubit id from channel id."""
    if "coupler" in ch:
        return int(ch.split("coupler_")[1].split("/")[0])
    return int(ch.split("/")[0])


def qubit_info(ch: ChannelId, hconfig: HamiltonianConfig) -> tuple[Qubit, int]:
    """Return qubit information for a given channel."""
    target = qubit_id(ch)
    return (hconfig.qubits[target], hconfig.hilbert_space_index(target))


def index(ch: ChannelId, hconfig: HamiltonianConfig) -> int:
    """Returns Hilbert space index from channel id."""
    return qubit_info(ch, hconfig)[1]


def _marginalize_probability(
    probabilities: NDArray,
    dims: list[int],
    measured_qubits: list[tuple[Qubit, int]],
    acquisition_type: AcquisitionType,
) -> NDArray:
    # probability dimensions are:
    # (*S, M, *H_dim)
    # discarding the last dimension since the results is just a scalar / tuple of scalar
    results_shape = np.moveaxis(probabilities, -2, 0).shape[:-1]
    # res has dimensions (M, *S)
    res = np.empty(results_shape)
    for measurement, (q, q_idx) in enumerate(measured_qubits):
        # select the total probability distribution of the specific measurement
        distribution = probabilities[..., measurement, :]
        # leading are the dimensions related to the sweepers
        leading = distribution.shape[:-1]
        # distributions now has dimensions (*S, H_1, H_2, ..., H_i)
        distribution = distribution.reshape(*leading, *dims)
        # selectig the axes (hence the subspace) corresponding to the measured qubit
        axes = tuple(len(leading) + i for i in range(len(dims)) if i != q_idx)
        # summing over all system
        # marginalized has dimensions (*S, H_i) where i is the qubit that's measured
        marginalized = distribution.sum(axis=axes)

        # Marginalized state is multiplied by the qubit's confusion matrix
        # before stacking because measured subsystems may have different dimensions.
        corrected_marginalized = np.einsum(
            "ij,...j",
            q.confusion_matrix,
            marginalized,
        )

        if acquisition_type is AcquisitionType.DISCRIMINATION:
            # now we group all the states >= 1, so we classify as 1
            corrected_marginalized = corrected_marginalized[..., 1:]
        else:
            # here we are element-wise multiplying the state probability vector
            # to the state number
            corrected_marginalized *= np.arange(0, q.transmon_levels, 1)

        res[measurement, ...] = corrected_marginalized.sum(axis=-1)

    return res


def add_confusion_matrix(
    qubit_list: list[Qubit], matrix_a: np.ndarray | None = None
) -> np.ndarray:
    """Function for applying single qubit confusion matrix from the set of qubit present in the system."""

    if matrix_a is None:
        matrix_a = qubit_list[0].confusion_matrix

    qubit_list.pop(0)
    if len(qubit_list) != 0:
        next_q = qubit_list[0]
        matrix_b = (
            next_q.confusion_matrix
            if next_q.confusion_matrix is not None
            else np.eye(next_q.transmon_levels)
        )
        matrix_a = add_confusion_matrix(qubit_list, np.kron(matrix_a, matrix_b))

    return matrix_a


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
) -> dict[PulseId, Result]:
    """Process measurement results from a cyclic quantum simulation, where the output for each measurement is
    excited state population.
    Computes readout results by projecting quantum state probabilities onto
    measurement subspaces and applying configured post-processing.
    """

    acq_id = list(acquisitions(sequence).keys())
    # from every acquisition pulse id we get the corresponding channel, and from the channel we get the
    # corresponding Hilbert-space index to marginalize.
    measured_qubits = [
        qubit_info(sequence.pulse_channels(ro_id)[0], hamiltonian) for ro_id in acq_id
    ]

    res = _marginalize_probability(
        probabilities=state_probs,
        dims=hamiltonian.dims,
        measured_qubits=measured_qubits,
        acquisition_type=options.acquisition_type,
    )

    # adding (fake) q component in the results for INTEGRATION acquisition
    if options.acquisition_type is AcquisitionType.INTEGRATION:
        res = np.random.normal(res, scale=0.001)
        zeros = np.zeros(res.shape) if np.ndim(res) != 0 else 0.0
        res = np.stack((res, zeros), axis=-1)

    # res is a (M, *S, ...) array
    return dict(zip(acq_id, res))


def _singleshot_results(
    state_probs: NDArray,
    sequence: PulseSequence,
    hamiltonian: HamiltonianConfig,
    options: ExecutionParameters,
) -> dict[PulseId, Result]:
    """Extract measurement results from simulated quantum state probabilities.
    Performs single-shot measurement extraction from state probabilities by sampling
    according to the specified number of shots and mapping readout operations to their
    corresponding measurement results.
    """

    # applying the confusion matrices to the density matrices
    state_probs = np.einsum(
        "ij,...j",
        add_confusion_matrix(list(hamiltonian.qubits.values())),
        state_probs,
    )
    # here we move the -2 index of the probability array, hence it now becomes:
    # (M, *S, *H_dim)
    state_probs = np.moveaxis(state_probs, -2, 0)

    # shots function returns a vector of shape:
    # (Nshots, M, *S, *H_dim)
    sampled = shots(state_probs, options.nshots)

    # move measurements dimension to the front, getting ready for extraction
    # the shape now is: (M, Nshots, *S, *H_dim)
    sampled = np.moveaxis(sampled, 1, 0)

    # from every acquisition pulse id we get the corresponding channel, and from the channel we get the
    # corresponding Hilbert-space index to extract from the sampled full-system state.
    acq_id = list(acquisitions(sequence).keys())
    qubit_indices = [
        index(sequence.pulse_channels(ro_id)[0], hamiltonian) for ro_id in acq_id
    ]

    # we use inverse_map to expand back the sampled results
    # res is a (M, M, Nshots, *S) array
    res = np.stack(np.unravel_index(sampled, hamiltonian.dims))[qubit_indices]

    # using np.einsum, so res is a (M, Nshots, *S, ...) array
    res = np.einsum(res, np.array([0, 0] + [...]), np.array([0] + [...]))

    if options.acquisition_type is AcquisitionType.DISCRIMINATION:
        # now we group all the states >= 1, so we classify as 1
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
    probabilities = _extract_probabilities(states)

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
