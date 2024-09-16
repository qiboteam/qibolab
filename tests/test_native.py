import contextlib

import numpy as np
import pytest
from pydantic import TypeAdapter

from qibolab._core.native import FixedSequenceFactory, RxyFactory, TwoQubitNatives
from qibolab._core.pulses import (
    Drag,
    Exponential,
    Gaussian,
    GaussianSquare,
    Pulse,
    Rectangular,
)
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
    factory = FixedSequenceFactory(seq)

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
        ({"theta": 7 * np.pi}, 1.0, 0.0),
        ({"theta": 7.5 * np.pi}, -0.5, 0.0),
        ({"phi": 7.5 * np.pi}, 1.0, 1.5 * np.pi),
    ],
)
def test_rxy_rotation_factory(args, amplitude, phase):
    seq = PulseSequence(
        [
            (
                "1/drive",
                Pulse(duration=40, amplitude=1.0, envelope=Gaussian(rel_sigma=3.0)),
            )
        ]
    )
    factory = RxyFactory(seq)

    fseq1 = factory.create_sequence(**args)
    fseq2 = factory.create_sequence(**args)
    assert fseq1 == fseq2
    np = "new/probe"
    fseq2.append((np, Pulse(duration=56, amplitude=0.43, envelope=Rectangular())))
    assert np not in fseq1.channels

    pulse = next(iter(fseq1.channel("1/drive")))
    assert pulse.amplitude == pytest.approx(amplitude)
    assert pulse.relative_phase == pytest.approx(phase)


def test_rxy_factory_multiple_channels():
    seq = PulseSequence(
        [
            (
                "1/drive",
                Pulse(duration=40, amplitude=0.7, envelope=Gaussian(rel_sigma=5.0)),
            ),
            (
                "2/drive",
                Pulse(duration=30, amplitude=1.0, envelope=Gaussian(rel_sigma=3.0)),
            ),
        ]
    )

    with pytest.raises(ValueError, match="Incompatible number of channels"):
        _ = RxyFactory(seq)


def test_rxy_factory_multiple_pulses():
    seq = PulseSequence(
        [
            (
                "1/drive",
                Pulse(duration=40, amplitude=0.08, envelope=Gaussian(rel_sigma=4.0)),
            ),
            (
                "1/drive",
                Pulse(duration=80, amplitude=0.76, envelope=Gaussian(rel_sigma=4.0)),
            ),
        ]
    )

    with pytest.raises(ValueError, match="Incompatible number of pulses"):
        _ = RxyFactory(seq)


@pytest.mark.parametrize(
    "envelope",
    [
        Gaussian(rel_sigma=3.0),
        GaussianSquare(rel_sigma=3.0, width=80),
        Drag(rel_sigma=5.0, beta=-0.037),
        Rectangular(),
        Exponential(tau=0.7, upsilon=0.8),
    ],
)
def test_rxy_rotation_factory_envelopes(envelope):
    seq = PulseSequence(
        [("1/drive", Pulse(duration=100, amplitude=1.0, envelope=envelope))]
    )

    if isinstance(envelope, (Gaussian, Drag)):
        context = contextlib.nullcontext()
    else:
        context = pytest.raises(ValueError, match="Incompatible pulse envelope")

    with context:
        _ = TypeAdapter(RxyFactory).validate_python(seq)


def test_two_qubit_natives_symmetric():
    natives = TwoQubitNatives(
        CZ=FixedSequenceFactory(PulseSequence()),
        CNOT=FixedSequenceFactory(PulseSequence()),
        iSWAP=FixedSequenceFactory(PulseSequence()),
    )
    assert natives.symmetric is False

    natives = TwoQubitNatives(
        CZ=FixedSequenceFactory(PulseSequence()),
        iSWAP=FixedSequenceFactory(PulseSequence()),
    )
    assert natives.symmetric is True

    natives = TwoQubitNatives(
        CZ=FixedSequenceFactory(PulseSequence()),
    )
    assert natives.symmetric is True

    natives = TwoQubitNatives(
        iSWAP=FixedSequenceFactory(PulseSequence()),
    )
    assert natives.symmetric is True

    natives = TwoQubitNatives(
        CNOT=FixedSequenceFactory(PulseSequence()),
    )
    assert natives.symmetric is False

    natives = TwoQubitNatives(
        CZ=FixedSequenceFactory(PulseSequence()),
        CNOT=FixedSequenceFactory(PulseSequence()),
    )
    assert natives.symmetric is False

    natives = TwoQubitNatives(
        CNOT=FixedSequenceFactory(PulseSequence()),
        iSWAP=FixedSequenceFactory(PulseSequence()),
    )
    assert natives.symmetric is False
