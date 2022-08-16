import pytest
import numpy as np
from qibo import gates
from qibo.backends import NumpyBackend
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence


def test_u3_sim_agreement():
    backend = NumpyBackend()
    theta, phi, lam = 0.1, 0.2, 0.3
    u3_matrix = gates.U3(0, theta, phi, lam).asmatrix(backend)
    rz1 = gates.RZ(0, phi).asmatrix(backend)
    rz2 = gates.RZ(0, theta).asmatrix(backend)
    rz3 = gates.RZ(0, lam).asmatrix(backend)
    rx1 = gates.RX(0, -np.pi / 2).asmatrix(backend)
    rx2 = gates.RX(0, np.pi / 2).asmatrix(backend)
    target_matrix = rz1 @ rx1 @ rz2 @ rx2 @ rz3
    np.testing.assert_allclose(u3_matrix, target_matrix)


def test_u3_to_sequence():
    platform = Platform("multiqubit")
    gate = gates.U3(0, theta=0.1, phi=0.2, lam=0.3)
    sequence = PulseSequence()
    platform.to_sequence(sequence, gate)
    assert len(sequence) == 2


def test_measurement():
    platform = Platform("multiqubit")
    gate = gates.M(0)
    with pytest.raises(NotImplementedError):
        platform.get_u3_parameters_from_gate(gate)
    sequence = PulseSequence()
    platform.to_sequence(sequence, gate)
    assert len(sequence) == 1
    assert len(sequence.qd_pulses) == 0
    assert len(sequence.qf_pulses) == 0
    assert len(sequence.ro_pulses) == 1


@pytest.mark.parametrize("gatename", ["H", "X", "Y", "Z"])
def test_pauli_to_u3_params(gatename):
    platform = Platform("multiqubit")
    gate = getattr(gates, gatename)(0)
    params = platform.get_u3_parameters_from_gate(gate)
    u3 = gates.U3(0, *params)
    if gatename in ("H", "Z"):
        np.testing.assert_allclose(gate.matrix, 1j * u3.matrix, atol=1e-15)
    else:
        np.testing.assert_allclose(gate.matrix, 1j * u3.matrix, atol=1e-15)


def test_identity_gate():
    platform = Platform("multiqubit")
    gate = gates.I(0)
    with pytest.raises(NotImplementedError):
        platform.get_u3_parameters_from_gate(gate)


@pytest.mark.parametrize("gatename", ["RX", "RY", "RZ"])
def test_rotations_to_u3_params(gatename):
    backend = NumpyBackend()
    platform = Platform("multiqubit")
    gate = getattr(gates, gatename)(0, theta=0.1)
    params = platform.get_u3_parameters_from_gate(gate)
    target_matrix = gates.U3(0, *params).asmatrix(backend)
    np.testing.assert_allclose(gate.asmatrix(backend), target_matrix)


def test_rz_to_sequence():
    platform = Platform("multiqubit")
    sequence = PulseSequence()
    platform.to_sequence(sequence, gates.RZ(0, theta=0.2))
    platform.to_sequence(sequence, gates.Z(0))
    assert len(sequence) == 0
    assert sequence.phase == 0.2 + np.pi


def test_u2_to_u3_params():
    backend = NumpyBackend()
    platform = Platform("multiqubit")
    gate = gates.U2(0, phi=0.1, lam=0.3)
    params = platform.get_u3_parameters_from_gate(gate)
    target_matrix = gates.U3(0, *params).asmatrix(backend)
    np.testing.assert_allclose(gate.asmatrix(backend), target_matrix)


def test_unitary_to_u3_params():
    from scipy.linalg import expm, det
    backend = NumpyBackend()
    platform = Platform("multiqubit")
    u = np.random.random((2, 2)) + 1j * np.random.random((2, 2))
    # make random matrix unitary
    u = expm(1j * (u + u.T.conj()))
    # transform to SU(2) form
    u = u / np.sqrt(det(u))
    gate = gates.Unitary(u, 0)
    params = platform.get_u3_parameters_from_gate(gate)
    target_matrix = gates.U3(0, *params).asmatrix(backend)
    np.testing.assert_allclose(gate.asmatrix(backend), target_matrix)


@pytest.mark.parametrize("platform_name", ['tiiq', 'qili', 'multiqubit']) #, 'icarusq'])
def test_pulse_sequence_add_u3(platform_name):
    platform = Platform(platform_name)
    seq = PulseSequence()
    platform.to_sequence(seq, gates.U3(0, 0.1, 0.2, 0.3))
    assert len(seq.pulses) == 2
    assert len(seq.qd_pulses) == 2

    RX90_pulse1 = platform.create_RX90_pulse(0, start = 0, phase = 0.3)
    RX90_pulse2 = platform.create_RX90_pulse(0, start = (RX90_pulse1.start + RX90_pulse1.duration), phase = 0.4 - np.pi)

    np.testing.assert_allclose(seq.time, RX90_pulse1.duration + RX90_pulse2.duration)
    np.testing.assert_allclose(seq.phase, 0.6)
    assert seq.serial == f"{RX90_pulse1.serial}, {RX90_pulse2.serial}"


@pytest.mark.parametrize("platform_name", ['tiiq', 'qili', 'multiqubit']) #, 'icarusq'])
def test_pulse_sequence_add_two_u3(platform_name):
    platform = Platform(platform_name)
    seq = PulseSequence()
    platform.to_sequence(seq, gates.U3(0, 0.1, 0.2, 0.3))
    platform.to_sequence(seq, gates.U3(0, 0.4, 0.6, 0.5))
    assert len(seq.pulses) == 4
    assert len(seq.qd_pulses) == 4

    RX90_pulse = platform.create_RX90_pulse(0)
    np.testing.assert_allclose(seq.phase, 0.6 + 1.5)
    np.testing.assert_allclose(seq.time, 2 * 2 * RX90_pulse.duration)

    RX90_pulse1 = platform.create_RX90_pulse(0, start = 0, phase = 0.3)
    RX90_pulse2 = platform.create_RX90_pulse(0, start = (RX90_pulse1.start + RX90_pulse1.duration), phase = 0.4 - np.pi)
    RX90_pulse3 = platform.create_RX90_pulse(0, start = (RX90_pulse2.start + RX90_pulse2.duration), phase = 1.1)
    RX90_pulse4 = platform.create_RX90_pulse(0, start = (RX90_pulse3.start + RX90_pulse3.duration), phase = 1.5 - np.pi)

    assert seq.serial == f"{RX90_pulse1.serial}, {RX90_pulse2.serial}, {RX90_pulse3.serial}, {RX90_pulse4.serial}"


@pytest.mark.parametrize("platform_name", ['tiiq', 'qili', 'multiqubit']) #, 'icarusq'])
def test_pulse_sequence_add_measurement(platform_name):
    platform = Platform(platform_name)
    seq = PulseSequence()
    platform.to_sequence(seq, gates.U3(0, 0.1, 0.2, 0.3))
    platform.to_sequence(seq, gates.M(0))
    assert len(seq.pulses) == 3
    assert len(seq.qd_pulses) == 2
    assert len(seq.ro_pulses) == 1

    np.testing.assert_allclose(seq.phase, 0.6)

    RX90_pulse1 = platform.create_RX90_pulse(0, start = 0, phase = 0.3)
    RX90_pulse2 = platform.create_RX90_pulse(0, start = RX90_pulse1.duration, phase = 0.4 - np.pi)
    MZ_pulse = platform.create_MZ_pulse(0, start = (RX90_pulse2.start + RX90_pulse2.duration), phase = 0.6)
    assert seq.serial == f"{RX90_pulse1.serial}, {RX90_pulse2.serial}, {MZ_pulse.serial}"
