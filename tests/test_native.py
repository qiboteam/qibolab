import numpy as np
import pytest

from qibolab._core.native import Native, SingleQubitNatives, TwoQubitNatives, rotation
from qibolab._core.pulses import Drag, Gaussian, Pulse, Rectangular
from qibolab._core.sequence import PulseSequence


def test_fixed_sequence_factory():
    seq = PulseSequence(
        [
            (
                "channel_1/probe",
                Pulse(duration=40, amplitude=0.3, envelope=Gaussian(rel_sigma=3.0)),
            ),
            (
                "channel_17/drive",
                Pulse(duration=125, amplitude=1.0, envelope=Rectangular()),
            ),
        ]
    )
    factory = Native(seq)

    fseq1 = factory.create_sequence()
    fseq2 = factory.create_sequence()
    assert fseq1 == seq
    assert fseq2 == seq
    # while representing the same sequence, the objects are actually different
    assert fseq1[0][1].id != fseq2[0][1].id

    np = "new/probe"
    fseq1.append(
        (
            np,
            Pulse(duration=30, amplitude=0.04, envelope=Drag(rel_sigma=4.0, beta=0.02)),
        )
    )
    assert np not in seq.channels
    assert np not in fseq2.channels

    # test alias
    assert factory() == seq


@pytest.mark.parametrize(
    "args,amplitude,phase",
    [
        ({}, 1.0, 0.0),
        ({"theta": np.pi / 2}, 0.5, 0.0),
        ({"phi": np.pi / 4}, 1.0, np.pi / 4),
        ({"theta": np.pi / 4, "phi": np.pi / 3}, 1.0 / 4, np.pi / 3),
        ({"theta": 3 * np.pi / 2}, -0.5, 0.0),
        ({"phi": 7.5 * np.pi}, 1.0, 1.5 * np.pi),
    ],
)
def test_rotation(args, amplitude, phase):
    seq = PulseSequence(
        [
            (
                "1/drive",
                Pulse(duration=40, amplitude=1.0, envelope=Gaussian(rel_sigma=3.0)),
            )
        ]
    )

    fseq1 = rotation(seq, **args)
    fseq2 = rotation(seq, **args)
    assert fseq1 == fseq2
    np = "new/probe"
    fseq2.append((np, Pulse(duration=56, amplitude=0.43, envelope=Rectangular())))
    assert np not in fseq1.channels

    pulse = next(iter(fseq1.channel("1/drive")))
    assert pulse.amplitude == pytest.approx(amplitude)
    assert pulse.relative_phase == pytest.approx(phase)


@pytest.mark.parametrize("amplitude", [0.4, 0.6])
@pytest.mark.parametrize("theta", [np.pi / 2, np.pi])
def test_rotation_rx90(amplitude, theta):
    """Testing rotation rx90 decomposition."""
    seq = PulseSequence(
        [
            (
                "1/drive",
                Pulse(
                    duration=40, amplitude=amplitude, envelope=Gaussian(rel_sigma=3.0)
                ),
            )
        ]
    )
    rx90_seq = rotation(seq, theta, phi=0, rx90=True)
    pulse_rx90 = next(iter(rx90_seq.channel("1/drive")))
    rx_seq = rotation(seq, theta, phi=0, rx90=False)
    pulse_rx = next(iter(rx_seq.channel("1/drive")))

    if theta * amplitude / np.pi > 0.5:
        assert len(rx90_seq) == 2
        assert pulse_rx90.amplitude == pulse_rx.amplitude
    else:
        assert len(rx90_seq) == 1
        assert pulse_rx90.amplitude == 2 * pulse_rx.amplitude


def test_two_qubit_natives_symmetric():
    natives = TwoQubitNatives(
        CZ=Native(PulseSequence()),
        CNOT=Native(PulseSequence()),
        iSWAP=Native(PulseSequence()),
    )
    assert natives.symmetric is False

    natives = TwoQubitNatives(
        CZ=Native(PulseSequence()),
        iSWAP=Native(PulseSequence()),
    )
    assert natives.symmetric is True

    natives = TwoQubitNatives(
        CZ=Native(PulseSequence()),
    )
    assert natives.symmetric is True

    natives = TwoQubitNatives(
        iSWAP=Native(PulseSequence()),
    )
    assert natives.symmetric is True

    natives = TwoQubitNatives(
        CNOT=Native(PulseSequence()),
    )
    assert natives.symmetric is False

    natives = TwoQubitNatives(
        CZ=Native(PulseSequence()),
        CNOT=Native(PulseSequence()),
    )
    assert natives.symmetric is False

    natives = TwoQubitNatives(
        CNOT=Native(PulseSequence()),
        iSWAP=Native(PulseSequence()),
    )
    assert natives.symmetric is False


def test_single_qubit_natives_r_uuid_uniqueness():
    """Test that multiple calls to SingleQubitNatives.R() return sequences with unique pulse UUIDs."""
    # Create a native gate
    seq = PulseSequence(
        [
            (
                "1/drive",
                Pulse(duration=40, amplitude=0.3, envelope=Gaussian(rel_sigma=3.0)),
            )
        ]
    )
    native_rx90 = Native(seq)

    # Test with RX90
    natives_rx90 = SingleQubitNatives(RX90=native_rx90)
    r1 = natives_rx90.R(theta=np.pi / 2, phi=0.0)
    r2 = natives_rx90.R(theta=np.pi / 2, phi=0.0)
    # Different calls should have unique UUIDs
    assert r1[0][1].id != r2[0][1].id

    # Test with different parameters
    r3 = natives_rx90.R(theta=np.pi, phi=np.pi / 4)
    assert r1[0][1].id != r3[0][1].id
    assert r2[0][1].id != r3[0][1].id

    # Test with RX
    native_rx = Native(seq)
    natives_rx = SingleQubitNatives(RX=native_rx)
    r4 = natives_rx.R(theta=np.pi / 2, phi=0.0)
    r5 = natives_rx.R(theta=np.pi / 2, phi=0.0)
    # Different calls should have unique UUIDs
    assert r4[0][1].id != r5[0][1].id

    # Test with different parameters
    r6 = natives_rx.R(theta=np.pi, phi=np.pi / 4)
    assert r4[0][1].id != r6[0][1].id
    assert r5[0][1].id != r6[0][1].id
