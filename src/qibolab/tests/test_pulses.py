"""Tests ``pulses.py``."""
import pytest
import numpy as np


def test_rectangular_shape():
    from qibolab.pulses import Pulse, Rectangular
    pulse = Pulse(start=0,
                    frequency=200_000_000,
                    amplitude=1,
                    duration=50,
                    phase=0,
                    shape='Rectangular()', 
                    channel=1)

    assert pulse.duration == 50
    assert isinstance(pulse.shape_object, Rectangular)
    assert pulse.shape_object.name == "Rectangular"
    assert repr(pulse.shape_object) == "Rectangular()"
    np.testing.assert_allclose(pulse.envelope_i, np.ones(50))
    np.testing.assert_allclose(pulse.envelope_q, np.zeros(50))


def test_gaussian_shape():
    from qibolab.pulses import Pulse, Gaussian
    duration = 50
    rel_sigma = 5
    pulse = Pulse(start=0,
                    frequency=200_000_000,
                    amplitude=1,
                    duration=duration,
                    phase=0,
                    shape=f'Gaussian({rel_sigma})', 
                    channel=1)

    assert pulse.duration == 50
    assert isinstance(pulse.shape_object, Gaussian)
    assert pulse.shape_object.name == "Gaussian"
    assert repr(pulse.shape_object) == f"Gaussian({float(rel_sigma)})"
    x = np.arange(0,duration,1)
    np.testing.assert_allclose(pulse.envelope_i, np.exp(-(1/2)*(((x-(duration-1)/2)**2)/(((duration)/rel_sigma)**2))))
    np.testing.assert_allclose(pulse.envelope_q, np.zeros(50))


def test_drag_shape():
    from qibolab.pulses import Pulse, Drag
    duration = 50
    rel_sigma = 5
    beta = 2
    pulse = Pulse(start=0,
                    frequency=200_000_000,
                    amplitude=1,
                    duration=duration,
                    phase=0,
                    shape=f'Drag({rel_sigma}, {beta})', 
                    channel=1)

    assert pulse.duration == 50
    assert isinstance(pulse.shape_object, Drag)
    assert pulse.shape_object.name == "Drag"
    assert repr(pulse.shape_object) == f"Drag({float(rel_sigma)}, {float(beta)})"
    x = np.arange(0,duration,1)
    np.testing.assert_allclose(pulse.envelope_i, np.exp(-(1/2)*(((x-(duration-1)/2)**2)/(((duration)/rel_sigma)**2))))
    np.testing.assert_allclose(pulse.envelope_q, beta * (-(x-(duration-1)/2)/((duration/rel_sigma)**2)) * np.exp(-(1/2)*(((x-(duration-1)/2)**2)/(((duration)/rel_sigma)**2))))


def test_pulse():
    from qibolab.pulses import Pulse
    duration = 50
    rel_sigma = 5
    beta = 2
    pulse = Pulse(start=0,
                    frequency=200_000_000,
                    amplitude=1,
                    duration=duration,
                    phase=0,
                    shape=f'Drag({rel_sigma}, {beta})', 
                    channel=1)

    target = f"Pulse(0, {duration}, 1.000, 200000000, 0.000, 'Drag({rel_sigma}, {beta})', 1, 'qd')"
    assert pulse.serial == target
    assert repr(pulse) == target


def test_readout_pulse():
    from qibolab.pulses import ReadoutPulse
    duration = 2000
    pulse = ReadoutPulse(start=0,
                        frequency=200_000_000,
                        amplitude=1,
                        duration=duration,
                        phase=0,
                        shape=f'Rectangular()', 
                        channel=11)

    target = f"ReadoutPulse(0, {duration}, 1.000, 200000000, 0.000, 'Rectangular()', 11, 'ro')"
    assert pulse.serial == target
    assert repr(pulse) == target
