"""
In this module we often recall, for the sake of a better interpretation of the whole pipeline, the transformed dimensions of the simulation's results tensor.
Generally this tensor will have the shape (*S, M *H_dim) (or its permutations),
where:
- *S is the number of iteration for each sweep in the experiment
- M is the number of measurements applied in the pulse sequence
- *H_dim is the list of subsystem dimensions
- |H_tot| is the dimension of the total system (i.e. Prod(*H_dim))

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
    hamiltonian: HamiltonianConfig,
    options: ExecutionParameters,
    qubit_to_m_map: dict[int, NDArray],
    sequence_qubit_arr: NDArray,
    **kwargs,
) -> Result:
    """Process measurement results from a cyclic quantum simulation, where the output for each measurement is
    excited state population.
    Computes readout results by projecting quantum state probabilities onto
    measurement subspaces and applying configured post-processing.
    """

    # now state_probs has shape (M_unique, *S, *H_dim)
    # dimensions of sweepers
    sweeps_dims = state_probs.shape[1 : -len(hamiltonian.dims)]

    # dimensions of total (not unique) measurement
    measurements_dim = sequence_qubit_arr.shape

    # shape is (M, *S), we are ignoring the dimensions of all subsystems
    result_shape = measurements_dim + sweeps_dims

    # non physical dimensions
    exp_axis = len(result_shape)
    # physical_dims
    total_axis = np.arange(len(hamiltonian.dims))

    res = np.empty(result_shape)
    for measured_q, unique_measure_idx in qubit_to_m_map.items():
        # marginal has shape (M_i, *S, H_i)
        # where M_i is the measurements on i-th transmon
        # and H_i is the Hilbert space dimension of it
        marginal = state_probs[unique_measure_idx, ...].sum(
            axis=tuple(total_axis[total_axis != measured_q] + exp_axis - 1)
        )

        if options.acquisition_type is AcquisitionType.DISCRIMINATION:
            # now we group all the states >= 1, so we classify as 1
            marginal = marginal[..., 1:]
        else:
            # here we are element-wise multiplying the state probability vector
            # to the state number
            marginal *= np.arange(0, marginal.shape[-1], 1)

        measured_q_idx = np.where(sequence_qubit_arr == measured_q)[0]

        res[measured_q_idx, ...] = marginal.sum(axis=-1)

    # adding (fake) q component in the results for INTEGRATION acquisition
    if options.acquisition_type is AcquisitionType.INTEGRATION:
        res = np.random.normal(res, scale=0.001)
        zeros = np.zeros(res.shape) if np.ndim(res) != 0 else 0.0
        res = np.stack((res, zeros), axis=-1)

    # res is a (M, *S, ...) array
    return res


def _singleshot_results(
    state_probs: NDArray,
    hamiltonian: HamiltonianConfig,
    options: ExecutionParameters,
    measurement_to_q_map: dict[int, NDArray],
    acquisition_inverse_map: NDArray,
    **kwargs,
) -> Result:
    """Extract measurement results from simulated quantum state probabilities.
    Performs single-shot measurement extraction from state probabilities by sampling
    according to the specified number of shots and mapping readout operations to their
    corresponding measurement results.
    """

    dim_array = np.asarray(hamiltonian.dims)

    # now sampled has shape (M_unique, *S, *H_dim)
    # dimensions of sweepers and shots
    sweeps_dims = state_probs.shape[1 : -len(dim_array)]

    # dimensions of total (not unique) measurement
    measurements_dim = tuple([sum(len(v) for v in measurement_to_q_map.values())])

    # dimensions of number of shots
    shots_dim = tuple([options.nshots])

    # shape is (M, Nshots *S), we are ignoring the dimensions of all subsystems
    result_shape = measurements_dim + shots_dim + sweeps_dims

    # non physical dimensions
    exp_axis = len(result_shape)
    # physical_dims
    total_axis = np.arange(len(dim_array))

    res = []
    for measure, unique_q_idx in measurement_to_q_map.items():
        # marginal has shape (*S, *H_mi)
        # results at the i-th unique measurement
        # and *H_mi is the list of all the dimensions of the measured transmons
        marginal = state_probs[measure, ...].sum(
            # 2 since we loose the M dimension and the axis starts from 0
            axis=tuple(total_axis[~np.isin(total_axis, unique_q_idx)] + exp_axis - 2),
        )

        # reshaping to (*S, |*H_mi|)
        marginal = marginal.reshape((*sweeps_dims, -1))

        # shots function returns a vector of shape:
        # (Nshots, *S, *H_mi)
        sampled = shots(marginal, options.nshots)

        # now marginal has dimensions (Q_i, Nshots, *S)
        marginal = np.stack(np.unravel_index(sampled, dim_array[unique_q_idx]))

        res.append(marginal)

    # stacking the results vertically
    # now it had dimensions (M, Nshot, *S)
    res = np.vstack(res)[acquisition_inverse_map]

    if options.acquisition_type is AcquisitionType.DISCRIMINATION:
        # now we group all the states >= 1, so we classify as 1
        res = np.clip(res, 0, 1)
    else:
        zeros = np.zeros(res.shape) if np.ndim(res) != 0 else 0.0
        res = np.stack((res, zeros), axis=-1)

    # res is a (M, Nshots *S, ...) array
    return res


def results(
    states: NDArray,
    sequence: PulseSequence,
    hamiltonian: HamiltonianConfig,
    options: ExecutionParameters,
) -> dict[PulseId, Result]:
    """Collect results for a single pulse sequence.

    The dictionary returned is already compliant with the expected
    result for the execution of this single sequence, thus suitable
    to be returned as is.
    """

    # probability dimensions are:
    # (*S, M, |H_tot|)
    probabilities = _extract_probabilities(states)
    # reshape probability vector, now is (*S, M, *H_dim)
    probabilities = probabilities.reshape(
        (*probabilities.shape[:-1], *hamiltonian.dims)
    )

    acq_id = list(acquisitions(sequence).keys())
    acq_t = np.asarray(list(acquisitions(sequence).values()))

    # select only unique times of measurements
    unique_acq_t, acq_direct_map, acq_inverse_map = np.unique(
        acq_t,
        return_index=True,
        return_inverse=True,
    )

    # order of qubits I measure during the sequence
    sequence_qubit_arr = np.asarray(
        [index(sequence.pulse_channels(ro_id)[0], hamiltonian) for ro_id in acq_id]
    )
    # select unique qubit indices of measured qubits
    unique_qubit_indices, qubit_inverse_map = np.unique(
        sequence_qubit_arr,
        return_inverse=True,
    )

    # mapping every measured qubit to the unique measurement index
    qubit_to_m_map = {
        uni_q: acq_inverse_map[qubit_inverse_map == n]
        for n, uni_q in enumerate(unique_qubit_indices)
    }

    # mapping every unique measurement to the related qubits
    measure_to_q_map = {
        n: unique_qubit_indices[qubit_inverse_map[acq_inverse_map == n]]
        for n in range(len(unique_acq_t))
    }

    confusion_matrices = []
    for i, qb in enumerate(hamiltonian.qubits.values()):
        # appending the confusion matrix
        confusion_matrices.append(qb.confusion_matrix)
        # appending the array index
        confusion_matrices.append([i + hamiltonian.nqubits, i])

    # applying confusion matrices to the probability vector
    # now corrected_probabilities ha dimensions (*S, M_unique, *H_dim)
    corrected_probabilities = np.einsum(
        *confusion_matrices,
        probabilities[acq_direct_map],
        [Ellipsis] + list(range(hamiltonian.nqubits)),
        [Ellipsis] + list(range(hamiltonian.nqubits, 2 * hamiltonian.nqubits)),
    )

    # now we reshuffle corrected_probabilities as (M_unique, *S, *H_dim)
    corrected_probabilities = np.moveaxis(
        corrected_probabilities, -(len(hamiltonian.dims) + 1), 0
    )

    results = (
        _singleshot_results
        if options.averaging_mode is AveragingMode.SINGLESHOT
        else _cyclic_results
    )

    return dict(
        zip(
            acq_id,
            results(
                state_probs=corrected_probabilities,
                hamiltonian=hamiltonian,
                options=options,
                qubit_to_m_map=qubit_to_m_map,  # used only in cyclic
                sequence_qubit_arr=sequence_qubit_arr,  # used only in cyclic
                measurement_to_q_map=measure_to_q_map,  # used only in singleshot
                acquisition_inverse_map=acq_inverse_map,  # used only in singleshot
            ),
        )
    )
