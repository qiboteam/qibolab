from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import QubitId, Result
from qibolab._core.pulses.pulse import Acquisition, Align, PulseId, Readout
from qibolab._core.sequence import PulseSequence

from .hamiltonians import HamiltonianConfig
from .operators import Operator


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
            if isinstance(ev, Align):
                raise ValueError("Align not supported in emulator.")
            time += ev.duration
    return acq


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
        tuple(hamiltonian.single_qubit),
        hamiltonian.nqubits,
        hamiltonian.transmon_levels,
    )

    assert options.nshots is not None
    sampled = shots(np.moveaxis(probabilities, -2, 0), options.nshots)
    # move measurements dimension to the front, getting ready for extraction
    measurements = np.moveaxis(sampled, 1, 0)

    results = {}
    # introduce cached measurements to avoid losing correlations
    cache_measurements = {}
    for (ro_id, sample), meas in zip(acquisitions(sequence).items(), measurements):
        qubit = int(sequence.pulse_channels(ro_id)[0].split("/")[0])
        cache_measurements.setdefault(sample, meas)
        assert hamiltonian.nqubits < 3, (
            "Results cannot be retrieved for more than 2 transmons"
        )
        res = (
            np.stack(
                divmod(cache_measurements[sample], hamiltonian.transmon_levels),
            )[qubit]
            if hamiltonian.nqubits == 2
            else cache_measurements[sample]
        )

        if options.acquisition_type is AcquisitionType.INTEGRATION:
            res = np.stack((res, np.zeros_like(res)), axis=-1)
            res = np.random.normal(res, scale=0.001)

        if options.averaging_mode == AveragingMode.CYCLIC:
            res = np.mean(res, axis=0)

        results[ro_id] = res

    return results
