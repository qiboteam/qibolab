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
    "gateargs",
    [
        (gates.I,),
        (gates.Z,),
        (gates.GPI, np.pi / 8),
        (gates.GPI2, -np.pi / 8),
        (gates.RZ, np.pi / 4),
    ],
)
def test_compile(platform, gateargs):
    nqubits = platform.nqubits
    circuit = generate_circuit_with_gate(nqubits, *gateargs)
    sequence = compile_circuit(circuit, platform)
    assert len(sequence.channels) == nqubits * int(gateargs[0] != gates.I) + nqubits


def test_compile_two_gates(platform):
    circuit = Circuit(1)
    circuit.add(gates.GPI2(0, phi=0.1))
    circuit.add(gates.GPI(0, 0.2))
    circuit.add(gates.M(0))

    sequence = compile_circuit(circuit, platform)

    qubit = platform.qubits[0]
    assert len(sequence.channels) == 2
    assert len(list(sequence.channel(qubit.drive.name))) == 2
    assert len(list(sequence.channel(qubit.probe.name))) == 2  # includes delay


def test_measurement(platform):
    nqubits = platform.nqubits
    circuit = Circuit(nqubits)
    qubits = [qubit for qubit in range(nqubits)]
    circuit.add(gates.M(*qubits))
    sequence = compile_circuit(circuit, platform)

    assert len(sequence.channels) == 1 * nqubits
    assert len(sequence.probe_pulses) == 1 * nqubits


def test_rz_to_sequence(platform):
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=0.2))
    circuit.add(gates.Z(0))
    sequence = compile_circuit(circuit, platform)
    assert len(sequence.channels) == 1
    assert len(sequence) == 2


def test_gpi_to_sequence(platform):
    circuit = Circuit(1)
    circuit.add(gates.GPI(0, phi=0.2))
    sequence = compile_circuit(circuit, platform)
    assert len(sequence.channels) == 1

    rx_seq = platform.qubits[0].native_gates.RX.create_sequence(phi=0.2)

    np.testing.assert_allclose(sequence.duration, rx_seq.duration)


def test_gpi2_to_sequence(platform):
    circuit = Circuit(1)
    circuit.add(gates.GPI2(0, phi=0.2))
    sequence = compile_circuit(circuit, platform)
    assert len(sequence.channels) == 1

    rx90_seq = platform.qubits[0].native_gates.RX.create_sequence(
        theta=np.pi / 2, phi=0.2
    )

    np.testing.assert_allclose(sequence.duration, rx90_seq.duration)
    assert sequence == rx90_seq


def test_cz_to_sequence():
    platform = create_platform("dummy")
    circuit = Circuit(3)
    circuit.add(gates.CZ(1, 2))

    sequence = compile_circuit(circuit, platform)
    test_sequence = platform.pairs[(2, 1)].native_gates.CZ.create_sequence()
    assert sequence[0] == test_sequence[0]


def test_cnot_to_sequence():
    platform = create_platform("dummy")
    circuit = Circuit(4)
    circuit.add(gates.CNOT(2, 3))

    sequence = compile_circuit(circuit, platform)
    test_sequence = platform.pairs[(2, 3)].native_gates.CNOT.create_sequence()
    assert sequence == test_sequence


def test_add_measurement_to_sequence(platform):
    circuit = Circuit(1)
    circuit.add(gates.GPI2(0, 0.1))
    circuit.add(gates.GPI2(0, 0.2))
    circuit.add(gates.M(0))

    sequence = compile_circuit(circuit, platform)
    qubit = platform.qubits[0]
    assert len(sequence.channels) == 2
    assert len(list(sequence.channel(qubit.drive.name))) == 2
    assert len(list(sequence.channel(qubit.probe.name))) == 2  # include delay

    s = PulseSequence()
    s.concatenate(qubit.native_gates.RX.create_sequence(theta=np.pi / 2, phi=0.1))
    s.concatenate(qubit.native_gates.RX.create_sequence(theta=np.pi / 2, phi=0.2))
    s.append((qubit.probe.name, Delay(duration=s.duration)))
    s.concatenate(qubit.native_gates.MZ.create_sequence())

    assert sequence == s


@pytest.mark.parametrize("delay", [0, 100])
def test_align_delay_measurement(platform, delay):
    circuit = Circuit(1)
    circuit.add(gates.Align(0, delay=delay))
    circuit.add(gates.M(0))
    sequence = compile_circuit(circuit, platform)

    target_sequence = PulseSequence()
    if delay > 0:
        target_sequence.append((platform.qubits[0].probe.name, Delay(duration=delay)))
    target_sequence.concatenate(platform.qubits[0].native_gates.MZ.create_sequence())
    assert sequence == target_sequence
    assert len(sequence.probe_pulses) == 1
