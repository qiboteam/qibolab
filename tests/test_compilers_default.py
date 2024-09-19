import numpy as np
import pytest
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.models import Circuit

from qibolab import create_platform
from qibolab._core.compilers import Compiler
from qibolab._core.identifier import ChannelId
from qibolab._core.native import FixedSequenceFactory, TwoQubitNatives
from qibolab._core.platform import Platform
from qibolab._core.pulses import Delay, Pulse
from qibolab._core.pulses.envelope import Rectangular
from qibolab._core.sequence import PulseSequence


def generate_circuit_with_gate(nqubits: int, gate, *params, **kwargs):
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


def compile_circuit(circuit: Circuit, platform: Platform) -> PulseSequence:
    """Compile a circuit to a pulse sequence."""
    compiler = Compiler.default()
    return compiler.compile(circuit, platform)[0]


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
def test_compile(platform: Platform, gateargs):
    nqubits = platform.nqubits
    circuit = generate_circuit_with_gate(nqubits, *gateargs)
    sequence = compile_circuit(circuit, platform)
    assert len(sequence.channels) == nqubits * int(gateargs[0] != gates.I) + nqubits * 1


def test_compile_two_gates(platform: Platform):
    circuit = Circuit(1)
    circuit.add(gates.GPI2(0, phi=0.1))
    circuit.add(gates.GPI(0, 0.2))
    circuit.add(gates.M(0))

    sequence = compile_circuit(circuit, platform)

    qubit = platform.qubits[0]
    assert len(sequence.channels) == 2
    assert len(list(sequence.channel(qubit.drive))) == 2
    assert len(list(sequence.channel(qubit.acquisition))) == 2  # includes delay


def test_measurement(platform: Platform):
    nqubits = platform.nqubits
    circuit = Circuit(nqubits)
    qubits = [qubit for qubit in range(nqubits)]
    circuit.add(gates.M(*qubits))
    sequence = compile_circuit(circuit, platform)

    assert len(sequence.channels) == 1 * nqubits
    assert len(sequence.acquisitions) == 1 * nqubits


def test_rz_to_sequence(platform: Platform):
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=0.2))
    circuit.add(gates.Z(0))
    sequence = compile_circuit(circuit, platform)
    assert len(sequence.channels) == 1
    assert len(sequence) == 2


def test_gpi_to_sequence(platform: Platform):
    natives = platform.natives

    circuit = Circuit(1)
    circuit.add(gates.GPI(0, phi=0.2))
    sequence = compile_circuit(circuit, platform)
    assert len(sequence.channels) == 1

    rx_seq = natives.single_qubit[0].RX.create_sequence(phi=0.2)

    np.testing.assert_allclose(sequence.duration, rx_seq.duration)


def test_gpi2_to_sequence(platform: Platform):
    natives = platform.natives

    circuit = Circuit(1)
    circuit.add(gates.GPI2(0, phi=0.2))
    sequence = compile_circuit(circuit, platform)
    assert len(sequence.channels) == 1

    rx90_seq = natives.single_qubit[0].RX.create_sequence(theta=np.pi / 2, phi=0.2)

    np.testing.assert_allclose(sequence.duration, rx90_seq.duration)
    assert sequence == rx90_seq


def test_cz_to_sequence():
    platform = create_platform("dummy")
    natives = platform.natives

    circuit = Circuit(3)
    circuit.add(gates.CZ(1, 2))

    sequence = compile_circuit(circuit, platform)
    test_sequence = natives.two_qubit[(2, 1)].CZ.create_sequence()
    assert sequence == test_sequence


def test_cnot_to_sequence():
    platform = create_platform("dummy")
    natives = platform.natives

    circuit = Circuit(4)
    circuit.add(gates.CNOT(2, 3))

    sequence = compile_circuit(circuit, platform)
    test_sequence = natives.two_qubit[(2, 3)].CNOT.create_sequence()
    assert sequence == test_sequence


def test_add_measurement_to_sequence(platform: Platform):
    natives = platform.natives

    circuit = Circuit(1)
    circuit.add(gates.GPI2(0, 0.1))
    circuit.add(gates.GPI2(0, 0.2))
    circuit.add(gates.M(0))

    sequence = compile_circuit(circuit, platform)
    qubit = platform.qubits[0]
    assert len(sequence.channels) == 2
    assert len(list(sequence.channel(qubit.drive))) == 2
    assert len(list(sequence.channel(qubit.acquisition))) == 2  # include delay

    s = PulseSequence()
    s.concatenate(natives.single_qubit[0].RX.create_sequence(theta=np.pi / 2, phi=0.1))
    s.concatenate(natives.single_qubit[0].RX.create_sequence(theta=np.pi / 2, phi=0.2))
    s.append((qubit.acquisition, Delay(duration=s.duration)))
    s.concatenate(natives.single_qubit[0].MZ.create_sequence())

    # the delay sorting depends on PulseSequence.channels, which is a set, and it's
    # order is not guaranteed
    def without_delays(seq: PulseSequence) -> PulseSequence:
        return [el for el in seq if not isinstance(el[1], Delay)]

    def delays(seq: PulseSequence) -> set[tuple[ChannelId, Delay]]:
        return {el for el in seq if isinstance(el[1], Delay)}

    assert without_delays(sequence) == without_delays(s)
    assert delays(sequence) == delays(s)


@pytest.mark.parametrize("delay", [0, 100])
def test_align_delay_measurement(platform: Platform, delay):
    natives = platform.natives

    circuit = Circuit(1)
    circuit.add(gates.Align(0, delay=delay))
    circuit.add(gates.M(0))
    sequence = compile_circuit(circuit, platform)

    target_sequence = PulseSequence()
    if delay > 0:
        target_sequence.append((platform.qubits[0].acquisition, Delay(duration=delay)))
    target_sequence.concatenate(natives.single_qubit[0].MZ.create_sequence())
    assert sequence == target_sequence
    assert len(sequence.acquisitions) == 1


def test_align_multiqubit(platform: Platform):
    main, coupled = 0, 2
    circuit = Circuit(3)
    circuit.add(gates.GPI2(main, phi=0.2))
    circuit.add(gates.CZ(main, coupled))
    circuit.add(gates.M(main, coupled))

    sequence = compile_circuit(circuit, platform)
    qubits = platform.qubits
    flux_duration = sequence.channel_duration(qubits[coupled].flux)
    for q in (main, coupled):
        probe_delay = next(iter(sequence.channel(qubits[q].acquisition)))
        assert isinstance(probe_delay, Delay)
        assert flux_duration == probe_delay.duration


@pytest.mark.parametrize("joint", [True, False])
def test_inactive_qubits(platform: Platform, joint: bool):
    main, coupled = 0, 1
    circuit = Circuit(2)
    circuit.add(gates.CZ(main, coupled))
    # another gate on drive is needed, to prevent trimming the delay, if alone
    circuit.add(gates.GPI2(coupled, phi=0.15))
    if joint:
        circuit.add(gates.M(main, coupled))
    else:
        circuit.add(gates.M(main))
        circuit.add(gates.M(coupled))

    natives = platform.natives.two_qubit[(main, coupled)] = TwoQubitNatives(
        CZ=FixedSequenceFactory([])
    )
    assert natives.CZ is not None
    natives.CZ.clear()
    sequence = compile_circuit(circuit, platform)

    qm = platform.qubit(main)[1]
    qc = platform.qubit(coupled)[1]
    readouts = {qm.probe, qm.acquisition, qc.probe, qc.acquisition}

    def no_measurement(seq: PulseSequence):
        return [el for el in seq if el[0] not in readouts]

    assert len(no_measurement(sequence)) == 1

    qubits = platform.qubits
    mflux = qubits[main].flux
    cdrive = qubits[coupled].drive
    duration = 200
    natives.CZ.extend(
        PulseSequence.load(
            [
                (
                    mflux,
                    Pulse(duration=duration, amplitude=0.42, envelope=Rectangular()),
                )
            ]
        )
    )
    padded_seq = compile_circuit(circuit, platform)
    assert len(no_measurement(padded_seq)) == 3
    cdrive_delay = next(iter(padded_seq.channel(cdrive)))
    assert isinstance(cdrive_delay, Delay)
    assert cdrive_delay.duration == next(iter(padded_seq.channel(mflux))).duration


def test_joint_split_equivalence(platform: Platform):
    """Test joint-split equivalence after 2q gate.

    Joint measurements are only equivalent to split in specific
    circumstances. When the two qubits involved are just coming out of a
    mutual interaction is one of those cases.

    Cf.
    https://github.com/qiboteam/qibolab/pull/992#issuecomment-2302708439
    """
    circuit = Circuit(3)
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.GPI2(2, phi=0.15))
    circuit.add(gates.CZ(0, 2))

    joint = Circuit(3)
    joint.add(gates.M(0, 2))

    joint_seq = compile_circuit(circuit + joint, platform)

    split = Circuit(3)
    split.add(gates.M(0))
    split.add(gates.M(2))

    split_seq = compile_circuit(circuit + split, platform)

    # the inter-channel sorting is unreliable, and mostly irrelevant (unless align
    # instructions are involved, which is not the case)
    assert not any(
        isinstance(p, gates.Align) for seq in (joint_seq, split_seq) for _, p in seq
    )  # TODO: gates.Align is just a placeholder, replace with the pulse-like when available
    qubits = platform.qubits
    for ch in (
        qubits[0].acquisition,
        qubits[2].acquisition,
        qubits[0].probe,
        qubits[2].probe,
    ):
        assert list(joint_seq.channel(ch)) == list(split_seq.channel(ch))
