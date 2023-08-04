import numpy as np
import pytest
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.models import Circuit

from qibolab.compilers import Compiler
from qibolab.pulses import PulseSequence
from qibolab.transpilers import Pipeline


def generate_circuit_with_gate(nqubits, gate, *params, **kwargs):
    circuit = Circuit(nqubits)
    circuit.add(gate(q, *params, **kwargs) for q in range(nqubits))
    circuit.add(gates.M(*range(nqubits)))
    return circuit


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


def compile_circuit(circuit, platform):
    """Compile a circuit to a pulse sequence."""
    transpiler = Pipeline.default(platform.two_qubit_native_types)
    compiler = Compiler.default()
    if transpiler.is_satisfied(circuit):
        native_circuit = circuit
    else:
        native_circuit, _ = transpiler(circuit)

    sequence, _ = compiler.compile(native_circuit, platform)
    return sequence


@pytest.mark.parametrize(
    "gateargs",
    [
        (gates.I,),
        (gates.X,),
        (gates.Y,),
        (gates.Z,),
        (gates.RX, np.pi / 8),
        (gates.RY, -np.pi / 8),
        (gates.RZ, np.pi / 4),
        (gates.U3, 0.1, 0.2, 0.3),
    ],
)
def test_compile(platform, gateargs):
    nqubits = platform.nqubits
    if gateargs[0] in (gates.I, gates.Z, gates.RZ):
        nseq = 0
    else:
        nseq = 2
    circuit = generate_circuit_with_gate(nqubits, *gateargs)
    sequence = compile_circuit(circuit, platform)
    assert len(sequence) == (nseq + 1) * nqubits


def test_compile_two_gates(platform):
    circuit = Circuit(1)
    circuit.add(gates.RX(0, theta=0.1))
    circuit.add(gates.RY(0, theta=0.2))
    circuit.add(gates.M(0))

    sequence = compile_circuit(circuit, platform)

    assert len(sequence.pulses) == 5
    assert len(sequence.qd_pulses) == 4
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
    assert len(sequence) == 0


def test_GPI2_to_sequence(platform):
    circuit = Circuit(1)
    circuit.add(gates.GPI2(0, phi=0.2))
    sequence = compile_circuit(circuit, platform)
    assert len(sequence.pulses) == 1
    assert len(sequence.qd_pulses) == 1

    RX90_pulse = platform.create_RX90_pulse(0, start=0, relative_phase=0.2)
    s = PulseSequence(RX90_pulse)

    np.testing.assert_allclose(sequence.duration, RX90_pulse.duration)
    assert sequence.serial == s.serial


def test_u3_to_sequence(platform):
    circuit = Circuit(1)
    circuit.add(gates.U3(0, 0.1, 0.2, 0.3))

    sequence = compile_circuit(circuit, platform)
    assert len(sequence.pulses) == 2
    assert len(sequence.qd_pulses) == 2

    RX90_pulse1 = platform.create_RX90_pulse(0, start=0, relative_phase=0.3)
    RX90_pulse2 = platform.create_RX90_pulse(0, start=RX90_pulse1.finish, relative_phase=0.4 - np.pi)
    s = PulseSequence(RX90_pulse1, RX90_pulse2)

    np.testing.assert_allclose(sequence.duration, RX90_pulse1.duration + RX90_pulse2.duration)
    assert sequence.serial == s.serial


def test_two_u3_to_sequence(platform):
    circuit = Circuit(1)
    circuit.add(gates.U3(0, 0.1, 0.2, 0.3))
    circuit.add(gates.U3(0, 0.4, 0.6, 0.5))

    sequence = compile_circuit(circuit, platform)
    assert len(sequence.pulses) == 4
    assert len(sequence.qd_pulses) == 4

    RX90_pulse = platform.create_RX90_pulse(0)

    np.testing.assert_allclose(sequence.duration, 2 * 2 * RX90_pulse.duration)

    RX90_pulse1 = platform.create_RX90_pulse(0, start=0, relative_phase=0.3)
    RX90_pulse2 = platform.create_RX90_pulse(0, start=RX90_pulse1.finish, relative_phase=0.4 - np.pi)
    RX90_pulse3 = platform.create_RX90_pulse(0, start=RX90_pulse2.finish, relative_phase=1.1)
    RX90_pulse4 = platform.create_RX90_pulse(0, start=RX90_pulse3.finish, relative_phase=1.5 - np.pi)
    s = PulseSequence(RX90_pulse1, RX90_pulse2, RX90_pulse3, RX90_pulse4)
    assert sequence.serial == s.serial


def test_cz_to_sequence(platform):
    if (1, 2) not in platform.pairs:
        pytest.skip(f"Skipping CZ test for {platform} because pair (1, 2) is not available.")

    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.CZ(0, 1))

    sequence = compile_circuit(circuit, platform)
    test_sequence, virtual_z_phases = platform.create_CZ_pulse_sequence((2, 1))
    assert len(sequence.pulses) == len(test_sequence) + 2


def test_add_measurement_to_sequence(platform):
    circuit = Circuit(1)
    circuit.add(gates.U3(0, 0.1, 0.2, 0.3))
    circuit.add(gates.M(0))

    sequence = compile_circuit(circuit, platform)
    assert len(sequence.pulses) == 3
    assert len(sequence.qd_pulses) == 2
    assert len(sequence.ro_pulses) == 1

    RX90_pulse1 = platform.create_RX90_pulse(0, start=0, relative_phase=0.3)
    RX90_pulse2 = platform.create_RX90_pulse(0, start=RX90_pulse1.finish, relative_phase=0.4 - np.pi)
    MZ_pulse = platform.create_MZ_pulse(0, start=RX90_pulse2.finish)
    s = PulseSequence(RX90_pulse1, RX90_pulse2, MZ_pulse)
    assert sequence.serial == s.serial
