import numpy as np
import pytest
from numpy.typing import NDArray

from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import Result
from qibolab._core.instruments.emulator.hamiltonians import HamiltonianConfig, Qubit
from qibolab._core.instruments.emulator.results import (
    acquisitions,
    calculate_probabilities_from_density_matrix,
    results,
    shots,
)
from qibolab._core.pulses.envelope import Rectangular
from qibolab._core.pulses.pulse import Acquisition, Pulse
from qibolab._core.sequence import PulseSequence


def _order_probabilities(probs, qubits):
    """Arrange probabilities according to the given `qubits ordering."""
    return np.transpose(
        probs, [i for i, _ in sorted(enumerate(qubits), key=lambda t: t[1])]
    )


def former_calculate(state, subsystems, nsubsystems, d):
    """Compute probabilities from density matrix."""
    order = tuple(sorted(subsystems))
    order += tuple(i for i in range(nsubsystems) if i not in subsystems)
    order = order + tuple(i + nsubsystems for i in order)

    shape = 2 * (d ** len(subsystems), d ** (nsubsystems - len(subsystems)))

    state = np.reshape(state, 2 * nsubsystems * (d,))
    state = np.reshape(np.transpose(state, order), shape)

    probs = np.abs(np.einsum("abab->a", state))
    probs = np.reshape(probs, len(subsystems) * (d,))

    return _order_probabilities(probs, subsystems).ravel()


def new_calculate(state, subsystems, nsubsystems, d):
    """Compute probabilities from density matrix."""
    state = np.reshape(state, 2 * nsubsystems * (d,))
    probs = np.abs(np.einsum(state, list(range(nsubsystems)) * 2, sorted(subsystems)))
    return _order_probabilities(probs, subsystems).ravel()


def former_apply_to_last_two_axes(func, array, *args, **kwargs):
    """Apply function over last two axes."""
    batch_shape = array.shape[:-2]
    m = array.shape[-1]
    reshaped_array = array.reshape(-1, m, m)
    processed = np.array([func(mat, *args, **kwargs) for mat in reshaped_array])
    return processed.reshape(*batch_shape, *processed.shape[1:])


def former_results(
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
        tuple(hamiltonian.qubits),
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
    for i, (ro_id, sample) in enumerate(acquisitions(sequence).items()):
        qubit = int(sequence.pulse_channels(ro_id)[0].split("/")[0])
        cache_measurements.setdefault(sample, measurements[i])
        assert hamiltonian.nqubits < 3, (
            "Results cannot be retrieved for more than 2 transmons"
        )
        res = (
            np.array(
                [
                    divmod(val, hamiltonian.transmon_levels)[qubit]
                    for val in cache_measurements[sample].flatten()
                ]
            ).reshape(measurements[i].shape)
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


def random_states(space: tuple[int, ...], sweeps: tuple[int, ...] = (), nacq: int = 1):
    dimension = np.prod(space, dtype=int)
    components = np.random.rand(*sweeps, nacq, dimension)

    state = components / np.sqrt((components**2).sum(axis=-1))[..., np.newaxis]

    return np.einsum("...i,...j->...ij", state, state)


def test_density_to_probs():
    density = random_states((3,) * 4)
    a = former_calculate(density, (1, 3), nsubsystems=4, d=3)
    b = new_calculate(density, (1, 3), nsubsystems=4, d=3)

    assert pytest.approx(a) == b


def test_apply_to_last_two_axes():
    densities = random_states((2,) * 4, (3, 2), nacq=2)
    a = former_apply_to_last_two_axes(
        new_calculate, densities, (1, 3), nsubsystems=4, d=2
    )
    b = calculate_probabilities_from_density_matrix(
        densities, (1, 3), nsubsystems=4, d=2
    )

    assert pytest.approx(a) == b


def test_resultz():
    densities = random_states((2,) * 2, (3, 2))
    sequence = PulseSequence(
        [("0/drive", Pulse(duration=20, amplitude=0.8, envelope=Rectangular()))]
    ) | PulseSequence([("0/acquisition", Acquisition(duration=1000))])
    hamiltonian = HamiltonianConfig(qubits={q: Qubit() for q in range(2)})
    options = ExecutionParameters(nshots=1000)

    fres = former_results(
        states=densities,
        sequence=sequence,
        hamiltonian=hamiltonian,
        options=options,
    )
    res = results(
        states=densities,
        sequence=sequence,
        hamiltonian=hamiltonian,
        options=options,
    )

    assert all(pytest.approx(r) == f for r, f in zip(res, fres))
