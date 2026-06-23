import numpy as np

from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.instruments.emulator.results import (
    _cyclic_results,
    _marginalize_probability,
)
from qibolab._core.pulses import Acquisition


def random_states(space: tuple[int, ...], sweeps: tuple[int, ...] = (), nacq: int = 1):
    dimension = np.prod(space, dtype=int)
    components = np.random.rand(*sweeps, nacq, dimension)

    state = components / np.sqrt((components**2).sum(axis=-1))[..., np.newaxis]

    return np.einsum("...i,...j->...ij", state, state)


def test_marginalize_probability_preserves_sweep_axes():
    probabilities = np.arange(1, 25).reshape(2, 12)

    marginalized = _marginalize_probability(probabilities, [2, 3, 2], 1)
    expected = probabilities.reshape(2, 2, 3, 2).sum(axis=(1, 3))

    np.testing.assert_allclose(marginalized, expected)


def test_cyclic_integration_results_marginalize_probabilities(monkeypatch):
    monkeypatch.setattr(np.random, "normal", lambda loc, scale: loc)
    acq0 = Acquisition(duration=1)
    acq1 = Acquisition(duration=1)
    sequence = _Sequence(
        {
            "0/acquisition": [acq0],
            "1/acquisition": [acq1],
        }
    )
    probabilities = np.array(
        [
            [[0.10, 0.20, 0.30], [0.05, 0.15, 0.20]],
            [[0.10, 0.20, 0.30], [0.05, 0.15, 0.20]],
        ]
    )
    results = _cyclic_results(
        state_probs=probabilities.reshape(2, -1),
        sequence=sequence,
        hamiltonian=_HamiltonianConfig(dims=[2, 3]),
        options=ExecutionParameters(
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
    )

    np.testing.assert_allclose(results[acq0.id], [0.40, 0.0])
    np.testing.assert_allclose(results[acq1.id], [0.85, 0.0])


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
    def __init__(self, dims):
        self.dims = dims

    def hilbert_space_index(self, target):
        return target
