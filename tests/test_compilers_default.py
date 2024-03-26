import numpy as np
import pytest
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.models import Circuit

from qibolab import create_platform
from qibolab.compilers import Compiler
from qibolab.pulses import Delay, PulseSequence


def generate_circuit_with_gate(nqubits, gate, *params, **kwargs):
    circuit = Circuit(nqubits)
    circuit.add(gate(q, *params, **kwargs) for q in range(nqubits))
    circuit.add(gates.M(*range(nqubits)))
    return circuit


def test_u3_sim_agreement():
    backend = NumpyBackend()
    theta, phi, lam = 0.1, 0.2, 0.3
    u3_matrix = gates.U3(0, theta, phi, lam).matrix(backend)
    rz1 = gates.RZ(0, phi).matrix(backend)
    rz2 = gates.RZ(0, theta).matrix(backend)
    rz3 = gates.RZ(0, lam).matrix(backend)
    rx1 = gates.RX(0, -np.pi / 2).matrix(backend)
    rx2 = gates.RX(0, np.pi / 2).matrix(backend)
    target_matrix = rz1 @ rx1 @ rz2 @ rx2 @ rz3
    np.testing.assert_allclose(u3_matrix, target_matrix)


def compile_circuit(circuit, platform):
    """Compile a circuit to a pulse sequence."""
    compiler = Compiler.default()
    sequence, _ = compiler.compile(circuit, platform)
    return sequence


@pytest.mark.parametrize(
    "gateargs,sequence_len",
    [
        ((gates.I,), 1),
        ((gates.Z,), 2),
        ((gates.GPI, np.pi / 8), 3),
        ((gates.GPI2, -np.pi / 8), 3),
        ((gates.RZ, np.pi / 4), 2),
        ((gates.U3, 0.1, 0.2, 0.3), 10),
    ],
)
def test_compile(platform, gateargs, sequence_len):
    nqubits = platform.nqubits
    circuit = generate_circuit_with_gate(nqubits, *gateargs)
    sequence = compile_circuit(circuit, platform)
    assert len(sequence) == nqubits * sequence_len


def test_compile_two_gates(platform):
    circuit = Circuit(1)
    circuit.add(gates.GPI2(0, phi=0.1))
    circuit.add(gates.U3(0, theta=0.1, phi=0.2, lam=0.3))
    circuit.add(gates.M(0))

    sequence = compile_circuit(circuit, platform)

    assert len(sequence) == 13
    assert len(sequence.qd_pulses) == 3
    assert len(sequence.ro_pulses) == 1


def test_measurement(platform):
    nqubits = platform.nqubits
    circuit = Circuit(nqubits)
    qubits = [qubit for qubit in range(nqubits)]
    circuit.add(gates.M(*qubits))
    sequence = compile_circuit(circuit, platform)

    assert len(sequence) == 1 * nqubits
    assert len(sequence.qd_pulses) == 0 * nqubits
    assert len(sequence.qf_pulses) == 0 * nqubits
    assert len(sequence.ro_pulses) == 1 * nqubits


def test_rz_to_sequence(platform):
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=0.2))
    circuit.add(gates.Z(0))
    sequence = compile_circuit(circuit, platform)
    assert len(sequence) == 2


def test_gpi_to_sequence(platform):
    circuit = Circuit(1)
    circuit.add(gates.GPI(0, phi=0.2))
    sequence = compile_circuit(circuit, platform)
    assert len(sequence) == 1
    assert len(sequence.qd_pulses) == 1

    rx_pulse = platform.create_RX_pulse(0, relative_phase=0.2)
    s = PulseSequence([rx_pulse])

    np.testing.assert_allclose(sequence.duration, rx_pulse.duration)


def test_gpi2_to_sequence(platform):
    circuit = Circuit(1)
    circuit.add(gates.GPI2(0, phi=0.2))
    sequence = compile_circuit(circuit, platform)
    assert len(sequence) == 1
    assert len(sequence.qd_pulses) == 1

    rx90_pulse = platform.create_RX90_pulse(0, relative_phase=0.2)
    s = PulseSequence([rx90_pulse])

    np.testing.assert_allclose(sequence.duration, rx90_pulse.duration)
    assert sequence == s


def test_u3_to_sequence(platform):
    circuit = Circuit(1)
    circuit.add(gates.U3(0, 0.1, 0.2, 0.3))

    sequence = compile_circuit(circuit, platform)
    assert len(sequence) == 8
    assert len(sequence.qd_pulses) == 2

    rx90_pulse1 = platform.create_RX90_pulse(0, relative_phase=0.3)
    rx90_pulse2 = platform.create_RX90_pulse(0, relative_phase=0.4 - np.pi)
    s = PulseSequence([rx90_pulse1, rx90_pulse2])

    np.testing.assert_allclose(
        sequence.duration, rx90_pulse1.duration + rx90_pulse2.duration
    )
    # assert sequence == s


def test_two_u3_to_sequence(platform):
    circuit = Circuit(1)
    circuit.add(gates.U3(0, 0.1, 0.2, 0.3))
    circuit.add(gates.U3(0, 0.4, 0.6, 0.5))

    sequence = compile_circuit(circuit, platform)
    assert len(sequence) == 18
    assert len(sequence.qd_pulses) == 4

    rx90_pulse = platform.create_RX90_pulse(0)

    np.testing.assert_allclose(sequence.duration, 2 * 2 * rx90_pulse.duration)

    rx90_pulse1 = platform.create_RX90_pulse(0, relative_phase=0.3)
    rx90_pulse2 = platform.create_RX90_pulse(0, relative_phase=0.4 - np.pi)
    rx90_pulse3 = platform.create_RX90_pulse(0, relative_phase=1.1)
    rx90_pulse4 = platform.create_RX90_pulse(0, relative_phase=1.5 - np.pi)
    s = PulseSequence([rx90_pulse1, rx90_pulse2, rx90_pulse3, rx90_pulse4])
    # assert sequence == s


def test_cz_to_sequence():
    platform = create_platform("dummy")
    circuit = Circuit(3)
    circuit.add(gates.CZ(1, 2))

    sequence = compile_circuit(circuit, platform)
    test_sequence = platform.create_CZ_pulse_sequence((2, 1))
    assert sequence[0] == test_sequence[0]


def test_cnot_to_sequence():
    platform = create_platform("dummy")
    circuit = Circuit(4)
    circuit.add(gates.CNOT(2, 3))

    sequence = compile_circuit(circuit, platform)
    test_sequence = platform.create_CNOT_pulse_sequence((2, 3))
    assert len(sequence) == len(test_sequence) + 1
    assert sequence[0] == test_sequence[0]


def test_add_measurement_to_sequence(platform):
    circuit = Circuit(1)
    circuit.add(gates.U3(0, 0.1, 0.2, 0.3))
    circuit.add(gates.M(0))

    sequence = compile_circuit(circuit, platform)
    assert len(sequence) == 10
    assert len(sequence.qd_pulses) == 2
    assert len(sequence.ro_pulses) == 1

    rx90_pulse1 = platform.create_RX90_pulse(0, relative_phase=0.3)
    rx90_pulse2 = platform.create_RX90_pulse(0, relative_phase=0.4 - np.pi)
    mz_pulse = platform.create_MZ_pulse(0)
    delay = 2 * rx90_pulse1.duration
    s = PulseSequence(
        [rx90_pulse1, rx90_pulse2, Delay(delay, mz_pulse.channel), mz_pulse]
    )
    # assert sequence == s


@pytest.mark.parametrize("delay", [0, 100])
def test_align_delay_measurement(platform, delay):
    circuit = Circuit(1)
    circuit.add(gates.Align(0, delay=delay))
    circuit.add(gates.M(0))
    sequence = compile_circuit(circuit, platform)

    mz_pulse = platform.create_MZ_pulse(0)
    target_sequence = PulseSequence()
    if delay > 0:
        target_sequence.append(Delay(delay, mz_pulse.channel))
    target_sequence.append(mz_pulse)
    assert sequence == target_sequence
    assert len(sequence.ro_pulses) == 1
