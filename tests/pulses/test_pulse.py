"""Tests ``pulses.py``."""

import numpy as np
import pytest

from qibolab.pulses import (
    BaseEnvelope,
    Custom,
    Drag,
    ECap,
    Gaussian,
    GaussianSquare,
    Iir,
    Pulse,
    PulseType,
    Rectangular,
    Snz,
)


def test_init():
    # standard initialisation
    p0 = Pulse(
        duration=50,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=0.0,
        envelope=Rectangular(),
        channel="0",
        type=PulseType.READOUT,
        qubit=0,
    )
    assert p0.relative_phase == 0.0

    p1 = Pulse(
        duration=50,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=0.0,
        envelope=Rectangular(),
        channel="0",
        type=PulseType.READOUT,
        qubit=0,
    )
    assert p1.type is PulseType.READOUT

    # initialisation with non int (float) frequency
    p2 = Pulse(
        duration=50,
        amplitude=0.9,
        frequency=int(20e6),
        relative_phase=0,
        envelope=Rectangular(),
        channel="0",
        type=PulseType.READOUT,
        qubit=0,
    )
    assert isinstance(p2.frequency, float) and p2.frequency == 20_000_000

    # initialisation with non float (int) relative_phase
    p3 = Pulse(
        duration=50,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=1.0,
        envelope=Rectangular(),
        channel="0",
        type=PulseType.READOUT,
        qubit=0,
    )
    assert isinstance(p3.relative_phase, float) and p3.relative_phase == 1.0

    # initialisation with str shape
    p4 = Pulse(
        duration=50,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=0,
        envelope=Rectangular(),
        channel="0",
        type=PulseType.READOUT,
        qubit=0,
    )
    assert isinstance(p4.envelope, Rectangular)

    # initialisation with str channel and str qubit
    p5 = Pulse(
        duration=50,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=0,
        envelope=Rectangular(),
        channel="channel0",
        type=PulseType.READOUT,
        qubit=0,
    )
    assert p5.qubit == 0

    # initialisation with different frequencies, shapes and types
    p6 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=-50e6,
        envelope=Rectangular(),
        relative_phase=0,
        type=PulseType.READOUT,
    )
    p7 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=0,
        envelope=Rectangular(),
        relative_phase=0,
        type=PulseType.FLUX,
        qubit=0,
    )
    p8 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=50e6,
        envelope=Gaussian(rel_sigma=0.2),
        relative_phase=0,
        type=PulseType.DRIVE,
        qubit=2,
    )
    p9 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=50e6,
        envelope=Drag(rel_sigma=0.2, beta=2),
        relative_phase=0,
        type=PulseType.DRIVE,
        qubit=200,
    )
    p10 = Pulse.flux(
        duration=40,
        amplitude=0.9,
        envelope=Iir(
            a=np.array([-1, 1]), b=np.array([-0.1, 0.1001]), target=Rectangular()
        ),
        channel="0",
        qubit=200,
    )
    p11 = Pulse.flux(
        duration=40,
        amplitude=0.9,
        envelope=Snz(t_idling=10, b_amplitude=0.5),
        channel="0",
        qubit=200,
    )
    p13 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=400e6,
        envelope=ECap(alpha=2),
        relative_phase=0,
        type=PulseType.DRIVE,
    )
    p14 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=50e6,
        envelope=GaussianSquare(rel_sigma=0.2, width=0.9),
        relative_phase=0,
        type=PulseType.READOUT,
        qubit=2,
    )

    # initialisation with float duration
    p12 = Pulse(
        duration=34.33,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=1,
        envelope=Rectangular(),
        channel="0",
        type=PulseType.READOUT,
        qubit=0,
    )
    assert isinstance(p12.duration, float)
    assert p12.duration == 34.33


def test_attributes():
    channel = "0"
    qubit = 0

    p10 = Pulse(
        duration=50,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=0.0,
        envelope=Rectangular(),
        channel=channel,
        qubit=qubit,
    )

    assert isinstance(p10.duration, float) and p10.duration == 50
    assert isinstance(p10.amplitude, float) and p10.amplitude == 0.9
    assert isinstance(p10.frequency, float) and p10.frequency == 20_000_000
    assert isinstance(p10.envelope, BaseEnvelope)
    assert isinstance(p10.channel, type(channel)) and p10.channel == channel
    assert isinstance(p10.qubit, type(qubit)) and p10.qubit == qubit


def test_aliases():
    rop = Pulse(
        duration=50,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=0.0,
        envelope=Rectangular(),
        type=PulseType.READOUT,
        channel="0",
        qubit=0,
    )
    assert rop.qubit == 0

    dp = Pulse(
        duration=2000,
        amplitude=0.9,
        frequency=200_000_000,
        relative_phase=0.0,
        envelope=Gaussian(rel_sigma=5),
        channel="0",
        qubit=0,
    )
    assert dp.amplitude == 0.9
    assert isinstance(dp.envelope, Gaussian)

    fp = Pulse.flux(
        duration=300, amplitude=0.9, envelope=Rectangular(), channel="0", qubit=0
    )
    assert fp.channel == "0"


def test_pulse():
    duration = 50
    rel_sigma = 5
    beta = 2
    pulse = Pulse(
        frequency=200_000_000,
        amplitude=1,
        duration=duration,
        relative_phase=0,
        envelope=Drag(rel_sigma=rel_sigma, beta=beta),
        channel="1",
    )

    assert pulse.duration == duration


def test_readout_pulse():
    duration = 2000
    pulse = Pulse(
        frequency=200_000_000,
        amplitude=1,
        duration=duration,
        relative_phase=0,
        envelope=Rectangular(),
        channel="11",
        type=PulseType.READOUT,
    )

    assert pulse.duration == duration


def test_envelope_waveform_i_q():
    envelope_i = np.cos(np.arange(0, 10, 0.01))
    envelope_q = np.sin(np.arange(0, 10, 0.01))
    custom_shape_pulse = Custom(i_=envelope_i, q_=envelope_q)
    pulse = Pulse(
        duration=1000,
        amplitude=1,
        frequency=10e6,
        relative_phase=0,
        envelope=Rectangular(),
        channel="1",
    )

    custom_shape_pulse.i_ = pulse.i(1)
    pulse.duration = 2000
    with pytest.raises(ValueError):
        custom_shape_pulse.i(samples=10)
    with pytest.raises(ValueError):
        custom_shape_pulse.q(samples=10)
