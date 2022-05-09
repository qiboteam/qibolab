"""Tests ``pulses.py`` and ``pulse_shapes.py``."""
import pytest
import numpy as np


def test_rectangular_shape():
    from qibolab.pulse_shapes import Rectangular
    rect = Rectangular()
    assert rect.name == "rectangular"
    np.testing.assert_allclose(rect.envelope(0.0, 0.2, 10.0, 4.5), 4.5 * np.ones(10))


def test_gaussian_shape():
    from qibolab.pulse_shapes import Gaussian
    gauss = Gaussian(1.5)
    assert gauss.name == "gaussian"
    assert gauss.rel_sigma == 1.5
    assert repr(gauss) == "(gaussian, 1.5)"
    envelope = gauss.envelope(0.0, 0.0, 10.0, 4.5)
    assert envelope.shape == (10,)


def test_drag_shape():
    from qibolab.pulse_shapes import Drag
    drag = Drag(1.5, 2.5)
    assert drag.name == "drag"
    assert drag.sigma == 1.5
    assert drag.beta == 2.5
    assert repr(drag) == "drag(1.5, 2.5)"
    target_envelop = 4.4108940298803985 + 1.470298009960133j
    time = np.array([1.0])
    assert drag.envelope(time, 0.2, 2.2, 4.5) == target_envelop


def test_swipht_shape():
    from qibolab.pulse_shapes import SWIPHT
    swipht = SWIPHT(2.2)
    assert swipht.name == "SWIPHT"
    assert swipht.g == 2.2
    assert repr(swipht) == "SWIPHT(2.2)"
    target_envelop = 4.4108940298803985
    time = np.array([1.0])
    assert swipht.envelope(time, 0.2, 2.2, 4.5) == 4.5


def test_pulse():
    from qibolab.pulses import Pulse
    from qibolab.pulse_shapes import Gaussian
    pulse = Pulse(0.0, 8.0, 0.8, 40.0, 0.7, Gaussian(1.0))
    target = "P(qcm, 0.0, 8.0, 0.8, 40.0, 0.7, (gaussian, 1.0))"
    assert pulse.serial() == target
    assert repr(pulse) == target
    assert pulse.compile().shape == (8,)


def test_readout_pulse():
    from qibolab.pulses import ReadoutPulse
    from qibolab.pulse_shapes import Rectangular
    pulse = ReadoutPulse(0.0, 8.0, 0.8, 40.0, 0.7, Rectangular())
    target = "P(qrm, 0.0, 8.0, 0.8, 40.0, 0.7, rectangular)"
    assert pulse.serial() == target
    assert repr(pulse) == target
    np.testing.assert_allclose(pulse.compile(), 0.8 * np.ones(8))
