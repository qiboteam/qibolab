import pytest
from qibo import K
from qibolab.states import HardwareState


def test_hardwarestate_init():
    state = HardwareState(1)
    with pytest.raises(NotImplementedError):
        state = HardwareState(2)


def test_hardwarestate_from_readout():
    state = HardwareState.from_readout((3 * 1e-4,), 250, 500)
    assert state.normalized_voltage == 0.2


def test_hardwarestate_copy():
    state = HardwareState.from_readout((3 * 1e-4,), 250, 500)
    state1 = state.copy(max_voltage=550)
    assert state1.normalized_voltage == 50 / 300
    state2 = state.copy(min_voltage=200)
    assert state2.normalized_voltage == 100 / 300


def test_hardwarestate_probabilities():
    state = HardwareState.from_readout((3 * 1e-4,), 250, 500)
    K.assert_allclose(state.probabilities(), [0.2, 0.8])