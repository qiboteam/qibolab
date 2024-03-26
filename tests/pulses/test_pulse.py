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
    PulseShape,
    PulseType,
    Rectangular,
    ShapeInitError,
    eCap,
)


def test_init():
    # standard initialisation
    p0 = Pulse(
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
    p6 = Pulse(40, 0.9, -50e6, 0, Rectangular(), 0, PulseType.READOUT)
    p7 = Pulse(40, 0.9, 0, 0, Rectangular(), 0, PulseType.FLUX, 0)
    p8 = Pulse(40, 0.9, 50e6, 0, Gaussian(5), 0, PulseType.DRIVE, 2)
    p9 = Pulse(40, 0.9, 50e6, 0, Drag(5, 2), 0, PulseType.DRIVE, 200)
    p10 = Pulse.flux(
        40, 0.9, IIR([-1, 1], [-0.1, 0.1001], Rectangular()), channel=0, qubit=200
    )
    p11 = Pulse.flux(40, 0.9, SNZ(t_idling=10, b_amplitude=0.5), channel=0, qubit=200)
    p13 = Pulse(40, 0.9, 400e6, 0, eCap(alpha=2), 0, PulseType.DRIVE)
    p14 = Pulse(40, 0.9, 50e6, 0, GaussianSquare(5, 0.9), 0, PulseType.READOUT, 2)

    # initialisation with float duration
    p12 = Pulse(
        duration=34.33,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=1,
        shape=Rectangular(),
        channel=0,
        type=PulseType.READOUT,
        qubit=0,
    )
    assert isinstance(p12.duration, float)
    assert p12.duration == 34.33


def test_attributes():
    channel = 0
    qubit = 0

    p10 = Pulse(
        duration=50,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=0.0,
        shape=Rectangular(),
        channel=channel,
        qubit=qubit,
    )

    assert type(p10.duration) == int and p10.duration == 50
    assert type(p10.amplitude) == float and p10.amplitude == 0.9
    assert type(p10.frequency) == int and p10.frequency == 20_000_000
    assert isinstance(p10.shape, PulseShape) and repr(p10.shape) == "Rectangular()"
    assert type(p10.channel) == type(channel) and p10.channel == channel
    assert type(p10.qubit) == type(qubit) and p10.qubit == qubit


def test_hash():
    rp = Pulse(40, 0.9, 100e6, 0, Rectangular(), 0, PulseType.DRIVE)
    dp = Pulse(40, 0.9, 100e6, 0, Drag(5, 1), 0, PulseType.DRIVE)
    hash(rp)
    my_dict = {rp: 1, dp: 2}
    assert list(my_dict.keys())[0] == rp
    assert list(my_dict.keys())[1] == dp

    p1 = Pulse(40, 0.9, 100e6, 0, Drag(5, 1), 0, PulseType.DRIVE)
    p2 = Pulse(40, 0.9, 100e6, 0, Drag(5, 1), 0, PulseType.DRIVE)

    assert p1 == p2

    p1 = Pulse(40, 0.9, 100e6, 0, Drag(5, 1), 0, PulseType.DRIVE)
    p2 = copy.copy(p1)
    p3 = copy.deepcopy(p1)
    assert p1 == p2
    assert p1 == p3


def test_aliases():
    rop = Pulse(
        duration=50,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=0.0,
        shape=Rectangular(),
        type=PulseType.READOUT,
        channel=0,
        qubit=0,
    )
    assert rop.qubit == 0

    dp = Pulse(
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
        duration=300, amplitude=0.9, shape=Rectangular(), channel=0, qubit=0
    )
    assert fp.channel == 0


def test_pulse():
    duration = 50
    rel_sigma = 5
    beta = 2
    pulse = Pulse(
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
