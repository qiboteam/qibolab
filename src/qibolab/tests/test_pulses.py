# -*- coding: utf-8 -*-
"""Tests ``pulses.py``."""
import numpy as np
import pytest

from qibolab.pulses import (
    Drag,
    Gaussian,
    Pulse,
    PulseSequence,
    ReadoutPulse,
    Rectangular,
)


def test_rectangular_shape():
    pulse = Pulse(
        start=0,
        frequency=200_000_000,
        amplitude=1,
        duration=50,
        phase=0,
        shape="Rectangular()",
        channel=1,
    )

    assert pulse.duration == 50
    assert isinstance(pulse.shape_object, Rectangular)
    assert pulse.shape_object.name == "Rectangular"
    assert repr(pulse.shape_object) == "Rectangular()"
    np.testing.assert_allclose(pulse.envelope_i, np.ones(50))
    np.testing.assert_allclose(pulse.envelope_q, np.zeros(50))


def test_gaussian_shape():
    duration = 50
    rel_sigma = 5
    pulse = Pulse(
        start=0,
        frequency=200_000_000,
        amplitude=1,
        duration=duration,
        phase=0,
        shape=f"Gaussian({rel_sigma})",
        channel=1,
    )

    assert pulse.duration == 50
    assert isinstance(pulse.shape_object, Gaussian)
    assert pulse.shape_object.name == "Gaussian"
    assert repr(pulse.shape_object) == f"Gaussian({float(rel_sigma)})"
    x = np.arange(0, duration, 1)
    np.testing.assert_allclose(
        pulse.envelope_i,
        np.exp(
            -(1 / 2)
            * (((x - (duration - 1) / 2) ** 2) / (((duration) / rel_sigma) ** 2))
        ),
    )
    np.testing.assert_allclose(pulse.envelope_q, np.zeros(50))


def test_drag_shape():
    duration = 50
    rel_sigma = 5
    beta = 2
    pulse = Pulse(
        start=0,
        frequency=200_000_000,
        amplitude=1,
        duration=duration,
        phase=0,
        shape=f"Drag({rel_sigma}, {beta})",
        channel=1,
    )

    assert pulse.duration == 50
    assert isinstance(pulse.shape_object, Drag)
    assert pulse.shape_object.name == "Drag"
    assert repr(pulse.shape_object) == f"Drag({float(rel_sigma)}, {float(beta)})"
    x = np.arange(0, duration, 1)
    np.testing.assert_allclose(
        pulse.envelope_i,
        np.exp(
            -(1 / 2)
            * (((x - (duration - 1) / 2) ** 2) / (((duration) / rel_sigma) ** 2))
        ),
    )
    np.testing.assert_allclose(
        pulse.envelope_q,
        beta
        * (-(x - (duration - 1) / 2) / ((duration / rel_sigma) ** 2))
        * np.exp(
            -(1 / 2)
            * (((x - (duration - 1) / 2) ** 2) / (((duration) / rel_sigma) ** 2))
        ),
    )


def test_pulse():
    duration = 50
    rel_sigma = 5
    beta = 2
    pulse = Pulse(
        start=0,
        frequency=200_000_000,
        amplitude=1,
        duration=duration,
        phase=0,
        shape=f"Drag({rel_sigma}, {beta})",
        channel=1,
    )

    target = f"Pulse(0, {duration}, 1.000, 200000000, 0.000, 'Drag({rel_sigma}, {beta})', 1, 'qd')"
    assert pulse.serial == target
    assert repr(pulse) == target


def test_readout_pulse():
    duration = 2000
    pulse = ReadoutPulse(
        start=0,
        frequency=200_000_000,
        amplitude=1,
        duration=duration,
        phase=0,
        shape=f"Rectangular()",
        channel=11,
    )

    target = f"ReadoutPulse(0, {duration}, 1.000, 200000000, 0.000, 'Rectangular()', 11, 'ro')"
    assert pulse.serial == target
    assert repr(pulse) == target


def test_pulse_sequence_add():
    sequence = PulseSequence()
    sequence.add(
        Pulse(
            start=0,
            frequency=200_000_000,
            amplitude=0.3,
            duration=60,
            phase=0,
            shape="Gaussian(5)",
            channel=1,
        )
    )
    sequence.add(
        Pulse(
            start=64,
            frequency=200_000_000,
            amplitude=0.3,
            duration=30,
            phase=0,
            shape="Gaussian(5)",
            channel=1,
        )
    )
    assert len(sequence.pulses) == 2
    assert len(sequence.qd_pulses) == 2


def test_pulse_sequence_add_readout():
    sequence = PulseSequence()
    sequence.add(
        Pulse(
            start=0,
            frequency=200_000_000,
            amplitude=0.3,
            duration=60,
            phase=0,
            shape="Gaussian(5)",
            channel=1,
        )
    )

    sequence.add(
        Pulse(
            start=64,
            frequency=200_000_000,
            amplitude=0.3,
            duration=60,
            phase=0,
            shape="Drag(5, 2)",
            channel=1,
            type="qf",
        )
    )

    sequence.add(
        ReadoutPulse(
            start=128,
            frequency=20_000_000,
            amplitude=0.9,
            duration=2000,
            phase=0,
            shape="Rectangular()",
            channel=11,
        )
    )
    assert len(sequence.pulses) == 3
    assert len(sequence.ro_pulses) == 1
    assert len(sequence.qd_pulses) == 1
    assert len(sequence.qf_pulses) == 1
