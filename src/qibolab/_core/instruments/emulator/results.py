from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from ...execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from ...identifier import ChannelId, QubitId, Result
from ...pulses import Acquisition, PulseId, Readout
from ...sequence import PulseSequence
from .engine import Operator
from .hamiltonians import HamiltonianConfig, Qubit

# DEBUG
import datetime

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


# def calculate_probabilities_from_density_matrix(
#     states: NDArray, subsystems: Iterable[QubitId], nsubsystems: int, d: int
# ) -> NDArray:
#     """Compute probabilities from density matrix."""
#     states_ = np.reshape(states, states.shape[:-2] + 2 * nsubsystems * (d,))
#     marginal = np.einsum(
#         states_,
#         # TODO: the `np.array()` wrapping call is only needed because of NumPy's type
#         # annotation - in practice, it also works without
#         np.array([...] + list(range(nsubsystems)) * 2),
#         np.array([...] + list(subsystems)),
#     )
#     return np.abs(marginal).reshape((*states.shape[:-2], -1))


def calculate_probabilities_from_density_matrix(
    states: NDArray
) -> NDArray:
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
) -> NDArray:
    """Select density matrices from states.

    First, retrieve acquisitions, and locate them in the tlist, to
    isolate the expectations related to measurements.

    The return type should be rank-3 array, where the last two are the density
    matrices dimensions, while the first one should correspond to the acquisitions.
    """
    acq = np.array(list(acquisitions))
    samples = np.minimum(np.searchsorted(times, acq), times.size - 1)
    return np.stack([states[n].full() for n in samples])


def add_noise_and_diff_acquisition(exp_data:np.ndarray, acquisition_type:AcquisitionType) -> np.ndarray:
    """Add Gaussian noise to experimental data and format it according to the acquisition type.

    In case of :const:`AcquisitionType.INTEGRATION` the data is formatted as if we are running a SIGNAL experiment on real hardware, 
    hence the single point is composed by the 2 IQ components; in the case of the emulator one component is simply null since all the information
    is simply carried by the magnitude of the signal.

    In case of :const:`AcquisitionType.DISCRIMINATION` the data is formatted as if we are running a PROBABILITY experiment on real hardware, 
    hence the single point is simply the 1-state probability, so we have to be sure that the added gaussian noise does not bring the computed value
    out of the probability definition interval 0 <= p <= 1.
    """

    np.random.seed(123456)
    exp_data = np.random.normal(exp_data, scale=0.001)

    if acquisition_type is AcquisitionType.INTEGRATION:
                zeros = np.zeros(exp_data.shape)
                exp_data = np.stack((exp_data, zeros), axis=-1)
            
    if acquisition_type is AcquisitionType.DISCRIMINATION:
        exp_data = np.clip(exp_data, 0, 1)

    return exp_data


def add_confusion_matrix(qubit_list: list[Qubit], matrix_a: np.ndarray | None = None) -> np.ndarray:
    """Function for applying single qubit confusion matrix from the set of qubit present in the system.
    """

    if matrix_a is None:
        matrix_a = qubit_list[0].confusion_matrix
    
    qubit_list.pop(0)
    if len(qubit_list) != 0:
        next_q = qubit_list[0]
        matrix_b = next_q.confusion_matrix if next_q.confusion_matrix is not None else np.eye(next_q.transmon_levels) 
        matrix_a = np.kron(matrix_a, matrix_b)
        add_confusion_matrix(qubit_list, matrix_a)
    
    return matrix_a


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
    probabilities = calculate_probabilities_from_density_matrix(
        states,
    )
    # apply the confusion matrix to the probability tensor
    # TODO: add also 2 qubit contributions to confusion matrix that spoils the tensor product
    probabilities = np.einsum('ij,...j', add_confusion_matrix(list(hamiltonian.qubits.values())), probabilities)

    results = {}

    if options.averaging_mode is AveragingMode.CYCLIC:

        states_computational_idx = np.stack(
                np.unravel_index([*range(probabilities.shape[-1])], hamiltonian.dims)
            )

        for ro_id in acquisitions(sequence).keys():
            i = index(sequence.pulse_channels(ro_id)[0], hamiltonian)

            res = np.sum(probabilities[..., i,states_computational_idx[i]==1],axis=-1)

            res = add_noise_and_diff_acquisition(res, options.acquisition_type)
            results[ro_id] = res


    if options.averaging_mode is AveragingMode.SINGLESHOT:

        assert options.nshots is not None
        sampled = shots(np.moveaxis(probabilities, -2, 0), options.nshots)
        # move measurements dimension to the front, getting ready for extraction
        measurements = np.moveaxis(sampled, 1, 0)
        # introduce cached measurements to avoid losing correlations
        cache_measurements = {}
        for (ro_id, sample), meas in zip(acquisitions(sequence).items(), measurements):
            i = index(sequence.pulse_channels(ro_id)[0], hamiltonian)
            cache_measurements.setdefault(sample, meas)
            res = np.stack(np.unravel_index(cache_measurements[sample], hamiltonian.dims))[
                i
            ]

            res = add_noise_and_diff_acquisition(res, options.acquisition_type)
            results[ro_id] = res

    # HERE I COULD RECOVER THE CORRECT SOLUTION ONLY FOR 'fixed-frequency-qutrits' PLATFORM
    # STILL NOT WORKING FOR 'qutrits' PLATFORM
    # list_res = [v for v in results.values()]
    # t = datetime.datetime.now().strftime("%H:%M:%S")
    # np.savez(f'{t}_platformpy_qutip_evolution.npz', np.stack(list_res))

    return results
