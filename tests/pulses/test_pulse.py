"""Tests ``pulses.py``."""

import numpy as np
import pytest

from qibolab.pulses import Acquisition, Custom, Pulse, Rectangular, VirtualZ
from qibolab.pulses.pulse import _Readout


def test_flux():
    p = Pulse.flux(duration=5, amplitude=0.9, envelope=Rectangular())
    assert p.relative_phase == 0

    p1 = Pulse.flux(duration=5, amplitude=0.9, relative_phase=1, envelope=Rectangular())
    assert p1.relative_phase == 0


def test_virtual_z():
    vz = VirtualZ(phase=-0.3)
    assert vz.duration == 0


def test_readout():
    p = Pulse(duration=5, amplitude=0.9, envelope=Rectangular())
    a = Acquisition(duration=60)
    r = _Readout(acquisition=a, probe=p)
    assert r.duration == a.duration
    assert r.id == a.id


def test_envelope_waveform_i_q():
    d = 1000
    p = Pulse(duration=d, amplitude=1, envelope=Rectangular())
    assert pytest.approx(p.i(1)) == np.ones(d)
    assert pytest.approx(p.i(2)) == np.ones(2 * d)
    assert pytest.approx(p.q(1)) == np.zeros(d)
    assert pytest.approx(p.envelopes(1)) == np.stack([np.ones(d), np.zeros(d)])

    envelope_i = np.cos(np.arange(0, 10, 0.01))
    envelope_q = np.sin(np.arange(0, 10, 0.01))
    custom_shape_pulse = Custom(i_=envelope_i, q_=envelope_q)
    pulse = Pulse(duration=1000, amplitude=1, relative_phase=0, envelope=Rectangular())

    custom_shape_pulse = custom_shape_pulse.model_copy(update={"i_": pulse.i(1)})
    with pytest.raises(ValueError):
        custom_shape_pulse.i(samples=10)
    with pytest.raises(ValueError):
        custom_shape_pulse.q(samples=10)
