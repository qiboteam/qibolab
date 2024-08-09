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
    Rectangular,
    Snz,
)


def test_init():
    # standard initialisation
    p0 = Pulse(
        duration=50,
        amplitude=0.9,
        relative_phase=0.0,
        envelope=Rectangular(),
    )
    assert p0.relative_phase == 0.0

    p1 = Pulse(
        duration=50,
        amplitude=0.9,
        relative_phase=0.0,
        envelope=Rectangular(),
    )
    assert p1.amplitude == 0.9

    # initialisation with non float (int) relative_phase
    p2 = Pulse(
        duration=50,
        amplitude=0.9,
        relative_phase=1.0,
        envelope=Rectangular(),
    )
    assert isinstance(p2.relative_phase, float) and p2.relative_phase == 1.0

    # initialisation with different shapes and types
    p6 = Pulse(
        duration=40,
        amplitude=0.9,
        envelope=Rectangular(),
        relative_phase=0,
    )
    p7 = Pulse(
        duration=40,
        amplitude=0.9,
        envelope=Rectangular(),
        relative_phase=0,
    )
    p8 = Pulse(
        duration=40,
        amplitude=0.9,
        envelope=Gaussian(rel_sigma=0.2),
        relative_phase=0,
    )
    p9 = Pulse(
        duration=40,
        amplitude=0.9,
        envelope=Drag(rel_sigma=0.2, beta=2),
        relative_phase=0,
    )
    p10 = Pulse.flux(
        duration=40,
        amplitude=0.9,
        envelope=Iir(
            a=np.array([-1, 1]), b=np.array([-0.1, 0.1001]), target=Rectangular()
        ),
    )
    p11 = Pulse.flux(
        duration=40,
        amplitude=0.9,
        envelope=Snz(t_idling=10, b_amplitude=0.5),
    )
    p13 = Pulse(
        duration=40,
        amplitude=0.9,
        envelope=ECap(alpha=2),
        relative_phase=0,
    )
    p14 = Pulse(
        duration=40,
        amplitude=0.9,
        envelope=GaussianSquare(rel_sigma=0.2, width=0.9),
        relative_phase=0,
    )

    # initialisation with float duration
    p12 = Pulse(
        duration=34.33,
        amplitude=0.9,
        relative_phase=1,
        envelope=Rectangular(),
    )
    assert isinstance(p12.duration, float)
    assert p12.duration == 34.33


def test_attributes():
    p = Pulse(
        duration=50,
        amplitude=0.9,
        relative_phase=0.0,
        envelope=Rectangular(),
    )

    assert isinstance(p.duration, float) and p.duration == 50
    assert isinstance(p.amplitude, float) and p.amplitude == 0.9
    assert isinstance(p.relative_phase, float) and p.relative_phase == 0.0
    assert isinstance(p.envelope, BaseEnvelope)


def test_pulse():
    duration = 50
    rel_sigma = 5
    beta = 2
    pulse = Pulse(
        amplitude=1,
        duration=duration,
        relative_phase=0,
        envelope=Drag(rel_sigma=rel_sigma, beta=beta),
    )

    assert pulse.duration == duration


def test_readout_pulse():
    duration = 2000
    pulse = Pulse(
        amplitude=1,
        duration=duration,
        relative_phase=0,
        envelope=Rectangular(),
    )

    assert pulse.duration == duration


def test_envelope_waveform_i_q():
    envelope_i = np.cos(np.arange(0, 10, 0.01))
    envelope_q = np.sin(np.arange(0, 10, 0.01))
    custom_shape_pulse = Custom(i_=envelope_i, q_=envelope_q)
    pulse = Pulse(
        duration=1000,
        amplitude=1,
        relative_phase=0,
        envelope=Rectangular(),
    )

    custom_shape_pulse = custom_shape_pulse.model_copy(update={"i_": pulse.i(1)})
    with pytest.raises(ValueError):
        custom_shape_pulse.i(samples=10)
    with pytest.raises(ValueError):
        custom_shape_pulse.q(samples=10)
