# -*- coding: utf-8 -*-
import numpy as np
import pytest
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.models import Circuit

from qibolab.backends import QibolabBackend
from qibolab.platform import Platform
from qibolab.platforms.abstract import AbstractPlatform
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


@pytest.mark.parametrize("platform_name", ["tii1q", "tii5q"])
def test_transpile(platform_name):
    backend = QibolabBackend(platform_name)
    platform: AbstractPlatform = backend.platform
    nqubits = platform.nqubits

    def generate_circuit_with_gate(gate, *params, **kwargs):
        _circuit = Circuit(nqubits)
        for qubit in range(nqubits):
            _circuit.add(gate(qubit, *params, **kwargs))
        qubits = [qubit for qubit in range(nqubits)]
        _circuit.add(gates.M(*qubits))
        return _circuit

    circuit = generate_circuit_with_gate(gates.I)
    sequence = platform.transpile(circuit)
    assert len(sequence) == (1 + 1) * nqubits

    circuit = generate_circuit_with_gate(gates.X)
    sequence = platform.transpile(circuit)
    assert len(sequence) == (1 + 1) * nqubits

    circuit = generate_circuit_with_gate(gates.Y)
    sequence = platform.transpile(circuit)
    assert len(sequence) == (1 + 1) * nqubits

    circuit = generate_circuit_with_gate(gates.Z)
    sequence = platform.transpile(circuit)
    assert len(sequence) == (0 + 1) * nqubits

    circuit = generate_circuit_with_gate(gates.RX, np.pi / 8)
    sequence = platform.transpile(circuit)
    assert len(sequence) == (1 + 1) * nqubits

    circuit = generate_circuit_with_gate(gates.RY, -np.pi / 8)
    sequence = platform.transpile(circuit)
    assert len(sequence) == (1 + 1) * nqubits

    circuit = generate_circuit_with_gate(gates.RZ, np.pi / 4)
    sequence = platform.transpile(circuit)
    assert len(sequence) == (0 + 1) * nqubits

    circuit = generate_circuit_with_gate(gates.U3, theta=0.1, phi=0.2, lam=0.3)
    sequence = platform.transpile(circuit)
    assert len(sequence) == (2 + 1) * nqubits

    circuit = Circuit(1)
    circuit.add(gates.RX(0, theta=0.1))
    circuit.add(gates.RY(0, theta=0.2))
    circuit.add(gates.M(0))

    sequence = platform.transpile(circuit)

    assert len(sequence.pulses) == 3
    assert len(sequence.qd_pulses) == 2
    assert len(sequence.ro_pulses) == 1

    RX_pulse = platform.create_RX_pulse(0)
    rotation_angle = 0.1
    RX_pulse.amplitude *= rotation_angle / np.pi
    RY_pulse = platform.create_RX_pulse(0, start=RX_pulse.finish, relative_phase=np.pi / 2)
    rotation_angle = 0.2
    RY_pulse.amplitude *= rotation_angle / np.pi
    MZ_pulse = platform.create_MZ_pulse(0, RY_pulse.finish)

    np.testing.assert_allclose(sequence.ro_pulses[0].start, MZ_pulse.start)
    np.testing.assert_allclose(sequence.ro_pulses[0].duration, MZ_pulse.duration)
    np.testing.assert_allclose(sequence.ro_pulses[0].amplitude, MZ_pulse.amplitude)
    np.testing.assert_allclose(sequence.ro_pulses[0].frequency, MZ_pulse.frequency)
    np.testing.assert_allclose(sequence.ro_pulses[0].phase, MZ_pulse.phase)
    np.testing.assert_allclose(sequence.qd_pulses[0].start, RX_pulse.start)
    np.testing.assert_allclose(sequence.qd_pulses[0].duration, RX_pulse.duration)
    np.testing.assert_allclose(sequence.qd_pulses[0].amplitude, RX_pulse.amplitude)
    np.testing.assert_allclose(sequence.qd_pulses[0].frequency, RX_pulse.frequency)
    np.testing.assert_allclose(sequence.qd_pulses[0].phase, RX_pulse.phase)
    np.testing.assert_allclose(sequence.qd_pulses[1].start, RY_pulse.start)
    np.testing.assert_allclose(sequence.qd_pulses[1].duration, RY_pulse.duration)
    np.testing.assert_allclose(sequence.qd_pulses[1].amplitude, RY_pulse.amplitude)
    np.testing.assert_allclose(sequence.qd_pulses[1].frequency, RY_pulse.frequency)
    np.testing.assert_allclose(sequence.qd_pulses[1].phase, RY_pulse.phase)


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
    from scipy.linalg import det, expm

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


@pytest.mark.parametrize("platform_name", ["tiiq", "qili", "multiqubit"])  # , 'icarusq'])
def test_pulse_sequence_add_u3(platform_name):
    platform = Platform(platform_name)
    seq = PulseSequence()
    platform.to_sequence(seq, gates.U3(0, 0.1, 0.2, 0.3))
    assert len(seq.pulses) == 2
    assert len(seq.qd_pulses) == 2

    RX90_pulse1 = platform.create_RX90_pulse(0, start=0, relative_phase=0.3)
    RX90_pulse2 = platform.create_RX90_pulse(0, start=(RX90_pulse1.start + RX90_pulse1.duration), phase=0.4 - np.pi)

    np.testing.assert_allclose(seq.time, RX90_pulse1.duration + RX90_pulse2.duration)
    np.testing.assert_allclose(seq.phase, 0.6)
    assert seq.serial == f"{RX90_pulse1.serial}, {RX90_pulse2.serial}"


@pytest.mark.parametrize("platform_name", ["tiiq", "qili", "multiqubit"])  # , 'icarusq'])
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

    RX90_pulse1 = platform.create_RX90_pulse(0, start=0, relative_phase=0.3)
    RX90_pulse2 = platform.create_RX90_pulse(
        0, start=(RX90_pulse1.start + RX90_pulse1.duration), relative_phase=0.4 - np.pi
    )
    RX90_pulse3 = platform.create_RX90_pulse(0, start=(RX90_pulse2.start + RX90_pulse2.duration), relative_phase=1.1)
    RX90_pulse4 = platform.create_RX90_pulse(
        0, start=(RX90_pulse3.start + RX90_pulse3.duration), relative_phase=1.5 - np.pi
    )

    assert seq.serial == f"{RX90_pulse1.serial}, {RX90_pulse2.serial}, {RX90_pulse3.serial}, {RX90_pulse4.serial}"


@pytest.mark.parametrize("platform_name", ["tiiq", "qili", "multiqubit"])  # , 'icarusq'])
def test_pulse_sequence_add_measurement(platform_name):
    platform = Platform(platform_name)
    seq = PulseSequence()
    platform.to_sequence(seq, gates.U3(0, 0.1, 0.2, 0.3))
    platform.to_sequence(seq, gates.M(0))
    assert len(seq.pulses) == 3
    assert len(seq.qd_pulses) == 2
    assert len(seq.ro_pulses) == 1

    np.testing.assert_allclose(seq.phase, 0.6)

    RX90_pulse1 = platform.create_RX90_pulse(0, start=0, relative_phase=0.3)
    RX90_pulse2 = platform.create_RX90_pulse(0, start=RX90_pulse1.duration, relative_phase=0.4 - np.pi)
    MZ_pulse = platform.create_MZ_pulse(0, start=(RX90_pulse2.start + RX90_pulse2.duration), relative_phase=0.6)
    assert seq.serial == f"{RX90_pulse1.serial}, {RX90_pulse2.serial}, {MZ_pulse.serial}"
