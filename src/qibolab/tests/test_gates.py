import pytest
import numpy as np
import qibo
from qibolab import gates
from qibo import K


def test_u3_to_sequence():
    from qibolab.circuit import PulseSequence
    qibo.set_backend("qibolab", platform="tiiq")
    gate = gates.U3(1, theta=0.1, phi=0.2, lam=0.3)
    sequence = PulseSequence()
    K.platform.add_u3_to_pulse_sequence(sequence, *gate.to_u3_params(), gate.target_qubits[0])
    assert len(sequence) == 2


def test_u3_sim_agreement():
    theta, phi, lam = 0.1, 0.2, 0.3 
    u3 = gates.U3(1, theta, phi, lam)
    rz1 = gates.RZ(1, phi).matrix
    rz2 = gates.RZ(1, theta).matrix
    rz3 = gates.RZ(1, lam).matrix
    rx1 = gates.RX(1, -np.pi / 2).matrix
    rx2 = gates.RX(1, np.pi / 2).matrix
    matrix = rz1 @ rx1 @ rz2 @ rx2 @ rz3
    np.testing.assert_allclose(matrix, u3.matrix)


def test_measurement():
    from qibolab.circuit import PulseSequence
    qibo.set_backend("qibolab", platform="tiiq")
    gate = gates.M(1)
    with pytest.raises(NotImplementedError):
        gate.to_u3_params()
    sequence = PulseSequence()
    K.platform.add_measurement_to_pulse_sequence(sequence, gate.target_qubits[0])
    assert len(sequence) == 1
    assert len(sequence.qd_pulses) == 0
    assert len(sequence.qf_pulses) == 0
    assert len(sequence.ro_pulses) == 1


@pytest.mark.parametrize("gatename", ["H", "X", "Y", "Z"])
def test_pauli_to_u3_params(gatename):
    gate = getattr(gates, gatename)(0)
    params = gate.to_u3_params()
    u3 = gates.U3(1, *params)
    if gatename in ("H", "Z"):
        np.testing.assert_allclose(gate.matrix, 1j * u3.matrix, atol=1e-15)
    else:
        np.testing.assert_allclose(gate.matrix, 1j * u3.matrix, atol=1e-15)


def test_identity_gate():
    from qibolab.circuit import PulseSequence
    gate = gates.I(1)
    with pytest.raises(NotImplementedError):
        gate.to_u3_params()


@pytest.mark.parametrize("gatename", ["RX", "RY", "RZ"])
def test_rotations_to_u3_params(gatename):
    gate = getattr(gates, gatename)(0, theta=0.1)
    params = gate.to_u3_params()
    u3 = gates.U3(1, *params)
    np.testing.assert_allclose(gate.matrix, u3.matrix)


def test_rz_to_sequence():
    from qibolab.circuit import PulseSequence
    gate = gates.RZ(1, theta=0.2)
    sequence = PulseSequence()
    sequence.phase += gate.parameters
    assert len(sequence) == 0
    assert sequence.phase == 0.2


def test_u2_to_u3_params():
    gate = gates.U2(1, phi=0.1, lam=0.3)
    params = gate.to_u3_params()
    u3 = gates.U3(1, *params)
    np.testing.assert_allclose(gate.matrix, u3.matrix)


def test_unitary_to_u3_params():
    from scipy.linalg import expm, det
    u = np.random.random((2, 2)) + 1j * np.random.random((2, 2))
    # make random matrix unitary
    u = expm(1j * (u + u.T.conj()))
    # transform to SU(2) form
    u = u / np.sqrt(det(u))
    gate = gates.Unitary(u, 0)
    params = gate.to_u3_params()
    u3 = gates.U3(1, *params)
    np.testing.assert_allclose(gate.matrix, u3.matrix)
