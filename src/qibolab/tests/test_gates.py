import pytest
import numpy as np
import qibo
from qibo import gates
from qibolab.circuit import PulseSequence


def test_u3_to_sequence():
    qibo.set_backend("qibolab")
    gate = gates.U3(0, theta=0.1, phi=0.2, lam=0.3)
    sequence = PulseSequence()
    gate.to_sequence(sequence)
    assert len(sequence) == 2


@pytest.mark.parametrize("gatename", ["H", "X", "Y", "Z"])
def test_pauli_to_u3_params(gatename):
    qibo.set_backend("qibolab")
    gate = getattr(gates, gatename)(0)
    params = gate.to_u3_params()
    u3 = gates.U3(0, *params)
    if gatename in ("H", "Z"):
        np.testing.assert_allclose(gate.matrix, 1j * u3.matrix, atol=1e-15)
    else:
        np.testing.assert_allclose(gate.matrix, 1j * u3.matrix, atol=1e-15)


@pytest.mark.parametrize("gatename", ["RX", "RY", "RZ"])
def test_rotations_to_u3_params(gatename):
    qibo.set_backend("qibolab")
    gate = getattr(gates, gatename)(0, theta=0.1)
    params = gate.to_u3_params()
    u3 = gates.U3(0, *params)
    np.testing.assert_allclose(gate.matrix, u3.matrix)


def test_rz_to_sequence():
    qibo.set_backend("qibolab")
    gate = gates.RZ(0, theta=0.2)
    sequence = PulseSequence()
    gate.to_sequence(sequence)
    assert len(sequence) == 0
    assert sequence.phase == 0.2


def test_u2_to_u3_params():
    qibo.set_backend("qibolab")
    gate = gates.U2(0, phi=0.1, lam=0.3)
    params = gate.to_u3_params()
    u3 = gates.U3(0, *params)
    np.testing.assert_allclose(gate.matrix, u3.matrix)