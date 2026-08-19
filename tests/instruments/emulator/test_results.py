import numpy as np
import pytest

import qibolab._core.instruments.emulator.results as result_module
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.instruments.emulator.hamiltonians import Qubit
from qibolab._core.instruments.emulator.results import results
from qibolab._core.pulses import Acquisition


def random_states(space: tuple[int, ...], sweeps: tuple[int, ...] = (), nacq: int = 1):
    dimension = np.prod(space, dtype=int)
    components = np.random.rand(*sweeps, nacq, dimension)

    state = components / np.sqrt((components**2).sum(axis=-1))[..., np.newaxis]

    return np.einsum("...i,...j->...ij", state, state)


def marginalization_case():
    acq = Acquisition(duration=1)
    sequence = _Sequence({"1/acquisition": [acq]})
    probabilities = np.array(
        [
            [0.10, 0.20, 0.10, 0.30, 0.20, 0.10],
            [0.20, 0.10, 0.20, 0.10, 0.30, 0.10],
        ]
    )
    states = np.stack([np.diag(probability) for probability in probabilities])[
        :, np.newaxis
    ]
    confusion_matrix = np.array(
        [
            [1.0, 0.1, 0.2],
            [0.0, 0.8, 0.3],
            [0.0, 0.1, 0.5],
        ]
    )
    hamiltonian = _HamiltonianConfig(
        dims=[2, 3],
        qubits={
            0: _UnmeasuredQubit(),
            1: Qubit(transmon_levels=3, confusion_matrix=confusion_matrix),
        },
    )

    return acq, sequence, states, hamiltonian


@pytest.mark.parametrize("average", [AveragingMode.CYCLIC, AveragingMode.SINGLESHOT])
def test_results(monkeypatch, average):
    monkeypatch.setattr(np.random, "normal", lambda loc, scale: loc)
    acq01 = Acquisition(duration=1)
    acq02 = Acquisition(duration=1)
    acq11 = Acquisition(duration=1)
    acq12 = Acquisition(duration=1)
    sequence = _Sequence(
        {
            "0/acquisition": [acq01, acq02],
            "1/acquisition": [acq11, acq12],
        }
    )
    rho = np.stack(
        [
            np.kron(
                np.array([[1, 0], [0, 0]]),
                np.array([[0.5, 0, 0], [0, 0.3, 0], [0, 0, 0.2]]),
            ),
            np.kron(
                np.array([[1, 0], [0, 0]]),
                np.array([[0.5, 0, 0], [0, 0.3, 0], [0, 0, 0.2]]),
            ),
            np.kron(
                np.array([[0.3, 0], [0, 0.7]]),
                np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            ),
            np.kron(
                np.array([[0.3, 0], [0, 0.7]]),
                np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            ),
        ]
    )
    res = results(
        states=rho,
        sequence=sequence,
        hamiltonian=_HamiltonianConfig(
            dims=[2, 3],
            qubits={
                0: Qubit(transmon_levels=2),
                1: Qubit(transmon_levels=3),
            },
        ),
        options=ExecutionParameters(
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=average,
            nshots=1e3,
        ),
    )

    if average is AveragingMode.CYCLIC:
        rtol = 1e-7
    else:
        rtol = 0.05

    # np,mean does nothing for CYCLIC sine is just a scalar
    prob01 = np.mean(res[acq01.id])
    prob02 = np.mean(res[acq02.id])
    prob11 = np.mean(res[acq11.id])
    prob12 = np.mean(res[acq12.id])

    np.testing.assert_allclose(prob01, 0, rtol=rtol)
    np.testing.assert_allclose(prob02, 0.7, rtol=rtol)
    np.testing.assert_allclose(prob11, 0.5, rtol=rtol)
    np.testing.assert_allclose(prob12, 1, rtol=rtol)


def test_confusion_matrix_applies_after_marginalization():
    acq, sequence, states, hamiltonian = marginalization_case()

    res = results(
        states=states,
        sequence=sequence,
        hamiltonian=hamiltonian,
        options=ExecutionParameters(
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
    )

    np.testing.assert_allclose(res[acq.id], [0.52, 0.60])


def test_singleshot_samples_confused_marginal_probabilities(monkeypatch):
    acq, sequence, states, hamiltonian = marginalization_case()
    captured = []

    def fake_shots(probabilities, nshots):
        captured.append(probabilities.copy())
        return np.zeros((int(nshots), *probabilities.shape[:-1]), dtype=int)

    monkeypatch.setattr(result_module, "shots", fake_shots)

    res = results(
        states=states,
        sequence=sequence,
        hamiltonian=hamiltonian,
        options=ExecutionParameters(
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.SINGLESHOT,
            nshots=5,
        ),
    )

    np.testing.assert_allclose(
        captured[0],
        [
            [0.48, 0.38, 0.14],
            [0.40, 0.41, 0.19],
        ],
    )
    assert res[acq.id].shape == (5, 2)


def test_singleshot_samples_joint_probabilities_for_simultaneous_measurements(
    monkeypatch,
):
    acq0 = Acquisition(duration=1)
    acq1 = Acquisition(duration=1)
    sequence = _Sequence({"0/acquisition": [acq0], "1/acquisition": [acq1]})
    probabilities = np.array([0.40, 0.10, 0.20, 0.30])
    states = np.diag(probabilities)[np.newaxis]
    captured = []

    def fake_shots(probabilities, nshots):
        captured.append(probabilities.copy())
        return np.array([0, 3, 1, 2, 3])

    monkeypatch.setattr(result_module, "shots", fake_shots)

    res = results(
        states=states,
        sequence=sequence,
        hamiltonian=_HamiltonianConfig(
            dims=[2, 2],
            qubits={
                0: Qubit(transmon_levels=2),
                1: Qubit(transmon_levels=2),
            },
        ),
        options=ExecutionParameters(
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.SINGLESHOT,
            nshots=5,
        ),
    )

    assert len(captured) == 1
    np.testing.assert_allclose(captured[0], probabilities)
    np.testing.assert_allclose(res[acq0.id], [0, 1, 0, 1, 1])
    np.testing.assert_allclose(res[acq1.id], [0, 1, 1, 0, 1])


class _UnmeasuredQubit:
    transmon_levels = 2

    @property
    def confusion_matrix(self):
        raise AssertionError("unmeasured qubit confusion matrix should not be used")


class _Sequence:
    def __init__(self, channels):
        self._channels = channels
        self.channels = list(channels)
        self._pulse_channels = {
            event.id: channel
            for channel, events in channels.items()
            for event in events
        }

    def channel(self, channel):
        return self._channels[channel]

    def pulse_channels(self, pulse_id):
        return [self._pulse_channels[pulse_id]]


class _HamiltonianConfig:
    def __init__(self, dims: list[int], qubits: dict[int, Qubit]):
        self.dims = dims
        self.qubits = qubits
        self.nqubits = len(qubits)

    def hilbert_space_index(self, target):
        return target
