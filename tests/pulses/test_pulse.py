"""Tests ``pulses.py``."""

import copy

import numpy as np
import pytest

from qibolab.pulses import (
    IIR,
    SNZ,
    Custom,
    Drag,
    Gaussian,
    GaussianSquare,
    Pulse,
    PulseSequence,
    PulseShape,
    PulseType,
    Rectangular,
    ShapeInitError,
    eCap,
)


def test_init():
    # standard initialisation
    p0 = Pulse(
        start=0,
        duration=50,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=0.0,
        shape=Rectangular(),
        channel=0,
        type=PulseType.READOUT,
        qubit=0,
    )
    assert p0.relative_phase == 0.0

    p1 = Pulse(
        start=100,
        duration=50,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=0.0,
        shape=Rectangular(),
        channel=0,
        type=PulseType.READOUT,
        qubit=0,
    )
    assert p1.type is PulseType.READOUT

    # initialisation with non int (float) frequency
    p2 = Pulse(
        start=0,
        duration=50,
        amplitude=0.9,
        frequency=int(20e6),
        relative_phase=0,
        shape=Rectangular(),
        channel=0,
        type=PulseType.READOUT,
        qubit=0,
    )
    assert isinstance(p2.frequency, int) and p2.frequency == 20_000_000

    # initialisation with non float (int) relative_phase
    p3 = Pulse(
        start=0,
        duration=50,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=1.0,
        shape=Rectangular(),
        channel=0,
        type=PulseType.READOUT,
        qubit=0,
    )
    assert isinstance(p3.relative_phase, float) and p3.relative_phase == 1.0

    # initialisation with str shape
    p4 = Pulse(
        start=0,
        duration=50,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=0,
        shape="Rectangular()",
        channel=0,
        type=PulseType.READOUT,
        qubit=0,
    )
    assert isinstance(p4.shape, Rectangular)

    # initialisation with str channel and str qubit
    p5 = Pulse(
        start=0,
        duration=50,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=0,
        shape="Rectangular()",
        channel="channel0",
        type=PulseType.READOUT,
        qubit="qubit0",
    )
    assert p5.qubit == "qubit0"

    # initialisation with different frequencies, shapes and types
    p6 = Pulse(0, 40, 0.9, -50e6, 0, Rectangular(), 0, PulseType.READOUT)
    p7 = Pulse(0, 40, 0.9, 0, 0, Rectangular(), 0, PulseType.FLUX, 0)
    p8 = Pulse(0, 40, 0.9, 50e6, 0, Gaussian(5), 0, PulseType.DRIVE, 2)
    p9 = Pulse(0, 40, 0.9, 50e6, 0, Drag(5, 2), 0, PulseType.DRIVE, 200)
    p10 = Pulse.flux(
        0, 40, 0.9, IIR([-1, 1], [-0.1, 0.1001], Rectangular()), channel=0, qubit=200
    )
    p11 = Pulse.flux(
        0, 40, 0.9, SNZ(t_idling=10, b_amplitude=0.5), channel=0, qubit=200
    )
    p13 = Pulse(0, 40, 0.9, 400e6, 0, eCap(alpha=2), 0, PulseType.DRIVE)
    p14 = Pulse(0, 40, 0.9, 50e6, 0, GaussianSquare(5, 0.9), 0, PulseType.READOUT, 2)

    # initialisation with float duration and start
    p12 = Pulse(
        start=5.5,
        duration=34.33,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=1,
        shape=Rectangular(),
        channel=0,
        type=PulseType.READOUT,
        qubit=0,
    )
    assert isinstance(p12.start, float)
    assert isinstance(p12.duration, float)
    assert p12.finish == 5.5 + 34.33


def test_attributes():
    channel = 0
    qubit = 0

    p10 = Pulse(
        start=10,
        duration=50,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=0.0,
        shape=Rectangular(),
        channel=channel,
        qubit=qubit,
    )

    assert type(p10.start) == int and p10.start == 10
    assert type(p10.duration) == int and p10.duration == 50
    assert type(p10.amplitude) == float and p10.amplitude == 0.9
    assert type(p10.frequency) == int and p10.frequency == 20_000_000
    assert type(p10.phase) == float and np.allclose(
        p10.phase, 2 * np.pi * p10.start * p10.frequency / 1e9
    )
    assert isinstance(p10.shape, PulseShape) and repr(p10.shape) == "Rectangular()"
    assert type(p10.channel) == type(channel) and p10.channel == channel
    assert type(p10.qubit) == type(qubit) and p10.qubit == qubit
    assert isinstance(p10.finish, int) and p10.finish == 60

    p0 = Pulse(
        start=0,
        duration=50,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=0.0,
        shape=Rectangular(),
        channel=0,
        type=PulseType.READOUT,
        qubit=0,
    )
    p0.start = 50
    assert p0.finish == 100


def test_is_equal_ignoring_start():
    """Checks if two pulses are equal, not looking at start time."""

    p1 = Pulse(0, 40, 0.9, 0, 0, Rectangular(), 0, PulseType.FLUX, 0)
    p2 = Pulse(100, 40, 0.9, 0, 0, Rectangular(), 0, PulseType.FLUX, 0)
    p3 = Pulse(0, 40, 0.9, 0, 0, Rectangular(), 0, PulseType.FLUX, 0)
    p4 = Pulse(200, 40, 0.9, 0, 0, Rectangular(), 2, PulseType.FLUX, 0)
    assert p1.is_equal_ignoring_start(p2)
    assert p1.is_equal_ignoring_start(p3)
    assert not p1.is_equal_ignoring_start(p4)

    p1 = Pulse(0, 40, 0.9, 50e6, 0, Gaussian(5), 0, PulseType.DRIVE, 2)
    p2 = Pulse(10, 40, 0.9, 50e6, 0, Gaussian(5), 0, PulseType.DRIVE, 2)
    p3 = Pulse(20, 50, 0.8, 50e6, 0, Gaussian(5), 0, PulseType.DRIVE, 2)
    p4 = Pulse(30, 40, 0.9, 50e6, 0, Gaussian(4), 0, PulseType.DRIVE, 2)
    assert p1.is_equal_ignoring_start(p2)
    assert not p1.is_equal_ignoring_start(p3)
    assert not p1.is_equal_ignoring_start(p4)


def test_hash():
    rp = Pulse(0, 40, 0.9, 100e6, 0, Rectangular(), 0, PulseType.DRIVE)
    dp = Pulse(0, 40, 0.9, 100e6, 0, Drag(5, 1), 0, PulseType.DRIVE)
    hash(rp)
    my_dict = {rp: 1, dp: 2}
    assert list(my_dict.keys())[0] == rp
    assert list(my_dict.keys())[1] == dp

    p1 = Pulse(0, 40, 0.9, 100e6, 0, Drag(5, 1), 0, PulseType.DRIVE)
    p2 = Pulse(0, 40, 0.9, 100e6, 0, Drag(5, 1), 0, PulseType.DRIVE)

    assert p1 == p2

    t0 = 0
    p1 = Pulse(t0, 40, 0.9, 100e6, 0, Drag(5, 1), 0, PulseType.DRIVE)
    p2 = copy.copy(p1)
    p3 = copy.deepcopy(p1)
    assert p1 == p2
    assert p1 == p3


def test_aliases():
    rop = Pulse(
        start=0,
        duration=50,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=0.0,
        shape=Rectangular(),
        type=PulseType.READOUT,
        channel=0,
        qubit=0,
    )
    assert rop.start == 0
    assert rop.qubit == 0

    dp = Pulse(
        start=0,
        duration=2000,
        amplitude=0.9,
        frequency=200_000_000,
        relative_phase=0.0,
        shape=Gaussian(5),
        channel=0,
        qubit=0,
    )
    assert dp.amplitude == 0.9
    assert isinstance(dp.shape, Gaussian)

    fp = Pulse.flux(
        start=0, duration=300, amplitude=0.9, shape=Rectangular(), channel=0, qubit=0
    )
    assert fp.channel == 0


def test_pulse_order():
    t0 = 0
    t = 0
    p1 = Pulse(t0, 400, 0.9, 20e6, 0, Gaussian(5), 10)
    p2 = Pulse(
        p1.finish + t,
        400,
        0.9,
        20e6,
        0,
        Rectangular(),
        qubit=30,
        type=PulseType.READOUT,
    )
    p3 = Pulse(p2.finish, 400, 0.9, 20e6, 0, Drag(5, 50), 20)
    ps1 = PulseSequence([p1, p2, p3])
    ps2 = PulseSequence([p3, p1, p2])

    def sortseq(sequence):
        return sorted(sequence, key=lambda item: (item.start, item.channel))

    assert sortseq(ps1) == sortseq(ps2)


def test_pulse():
    duration = 50
    rel_sigma = 5
    beta = 2
    pulse = Pulse(
        start=0,
        frequency=200_000_000,
        amplitude=1,
        duration=duration,
        relative_phase=0,
        shape=f"Drag({rel_sigma}, {beta})",
        channel=1,
    )

    assert pulse.duration == duration


def test_readout_pulse():
    duration = 2000
    pulse = Pulse(
        start=0,
        frequency=200_000_000,
        amplitude=1,
        duration=duration,
        relative_phase=0,
        shape=f"Rectangular()",
        channel=11,
        type=PulseType.READOUT,
    )

    assert pulse.duration == duration


def test_envelope_waveform_i_q():
    envelope_i = np.cos(np.arange(0, 10, 0.01))
    envelope_q = np.sin(np.arange(0, 10, 0.01))
    custom_shape_pulse = Custom(envelope_i, envelope_q)
    custom_shape_pulse_old_behaviour = Custom(envelope_i)
    pulse = Pulse(
        start=0,
        duration=1000,
        amplitude=1,
        frequency=10e6,
        relative_phase=0,
        shape="Rectangular()",
        channel=1,
    )

    with pytest.raises(ShapeInitError):
        custom_shape_pulse.envelope_waveform_i()
    with pytest.raises(ShapeInitError):
        custom_shape_pulse.envelope_waveform_q()

    custom_shape_pulse.pulse = pulse
    custom_shape_pulse_old_behaviour.pulse = pulse
    pulse.duration = 2000
    with pytest.raises(ValueError):
        custom_shape_pulse.pulse = pulse
        custom_shape_pulse.envelope_waveform_i()
    with pytest.raises(ValueError):
        custom_shape_pulse.pulse = pulse
        custom_shape_pulse.envelope_waveform_q()
