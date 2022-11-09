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


def test_transpile(platform_name):
    platform: AbstractPlatform = Platform(platform_name)
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
    assert len(sequence) == (0 + 1) * nqubits

    circuit = generate_circuit_with_gate(gates.X)
    sequence = platform.transpile(circuit)
    assert len(sequence) == (2 + 1) * nqubits

    circuit = generate_circuit_with_gate(gates.Y)
    sequence = platform.transpile(circuit)
    assert len(sequence) == (2 + 1) * nqubits

    circuit = generate_circuit_with_gate(gates.Z)
    sequence = platform.transpile(circuit)
    assert len(sequence) == (0 + 1) * nqubits

    circuit = generate_circuit_with_gate(gates.RX, np.pi / 8)
    sequence = platform.transpile(circuit)
    assert len(sequence) == (2 + 1) * nqubits

    circuit = generate_circuit_with_gate(gates.RY, -np.pi / 8)
    sequence = platform.transpile(circuit)
    assert len(sequence) == (2 + 1) * nqubits

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

    assert len(sequence.pulses) == 5
    assert len(sequence.qd_pulses) == 4
    assert len(sequence.ro_pulses) == 1


def test_measurement(platform_name):
    platform: AbstractPlatform = Platform(platform_name)
    nqubits = platform.nqubits
    circuit = Circuit(nqubits)
    qubits = [qubit for qubit in range(nqubits)]
    circuit.add(gates.M(*qubits))
    sequence: PulseSequence = platform.transpile(circuit)

    assert len(sequence) == 1 * nqubits
    assert len(sequence.qd_pulses) == 0 * nqubits
    assert len(sequence.qf_pulses) == 0 * nqubits
    assert len(sequence.ro_pulses) == 1 * nqubits


def test_rz_to_sequence(platform_name):
    platform = Platform(platform_name)
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=0.2))
    circuit.add(gates.Z(0))
    sequence: PulseSequence = platform.transpile(circuit)
    assert len(sequence) == 0
    assert sequence.virtual_z_phases[0] == 0.2 + np.pi


def test_u3_to_sequence(platform_name):
    platform = Platform(platform_name)
    circuit = Circuit(1)
    circuit.add(gates.U3(0, 0.1, 0.2, 0.3))

    sequence: PulseSequence = platform.transpile(circuit)
    assert len(sequence.pulses) == 2
    assert len(sequence.qd_pulses) == 2

    RX90_pulse1 = platform.create_RX90_pulse(0, start=0, relative_phase=0.3)
    RX90_pulse2 = platform.create_RX90_pulse(0, start=RX90_pulse1.finish, relative_phase=0.4 - np.pi)
    s = PulseSequence(RX90_pulse1, RX90_pulse2)

    np.testing.assert_allclose(sequence.duration, RX90_pulse1.duration + RX90_pulse2.duration)
    np.testing.assert_allclose(sequence.virtual_z_phases[0], 0.6)
    assert sequence.serial == s.serial


def test_two_u3_to_sequence(platform_name):
    platform = Platform(platform_name)
    circuit = Circuit(1)
    circuit.add(gates.U3(0, 0.1, 0.2, 0.3))
    circuit.add(gates.U3(0, 0.4, 0.6, 0.5))

    sequence: PulseSequence = platform.transpile(circuit)
    assert len(sequence.pulses) == 4
    assert len(sequence.qd_pulses) == 4

    RX90_pulse = platform.create_RX90_pulse(0)

    np.testing.assert_allclose(sequence.virtual_z_phases[0], 0.6 + 1.5)
    np.testing.assert_allclose(sequence.duration, 2 * 2 * RX90_pulse.duration)

    RX90_pulse1 = platform.create_RX90_pulse(0, start=0, relative_phase=0.3)
    RX90_pulse2 = platform.create_RX90_pulse(0, start=RX90_pulse1.finish, relative_phase=0.4 - np.pi)
    RX90_pulse3 = platform.create_RX90_pulse(0, start=RX90_pulse2.finish, relative_phase=1.1)
    RX90_pulse4 = platform.create_RX90_pulse(0, start=RX90_pulse3.finish, relative_phase=1.5 - np.pi)
    s = PulseSequence(RX90_pulse1, RX90_pulse2, RX90_pulse3, RX90_pulse4)
    assert sequence.serial == s.serial


def test_add_measurement_to_sequence(platform_name):
    platform = Platform(platform_name)
    circuit = Circuit(1)
    circuit.add(gates.U3(0, 0.1, 0.2, 0.3))
    circuit.add(gates.M(0))

    sequence: PulseSequence = platform.transpile(circuit)
    assert len(sequence.pulses) == 3
    assert len(sequence.qd_pulses) == 2
    assert len(sequence.ro_pulses) == 1

    np.testing.assert_allclose(sequence.virtual_z_phases[0], 0.6)

    RX90_pulse1 = platform.create_RX90_pulse(0, start=0, relative_phase=0.3)
    RX90_pulse2 = platform.create_RX90_pulse(0, start=RX90_pulse1.finish, relative_phase=0.4 - np.pi)
    MZ_pulse = platform.create_MZ_pulse(0, start=RX90_pulse2.finish)
    s = PulseSequence(RX90_pulse1, RX90_pulse2, MZ_pulse)
    assert sequence.serial == s.serial
