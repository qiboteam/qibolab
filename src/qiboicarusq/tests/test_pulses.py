import pytest
import numpy as np
from qiboicarusq import pulses
from qiboicarusq.circuit import PulseSequence


def test_basic_pulse():
    basic = pulses.BasicPulse(0, 0.5, 1.5, 0.8, 40.00, 0.7, "Rectangular")
    target_repr = "P(0, 0.5, 1.5, 0.8, 40.0, 0.7, Rectangular)"
    assert repr(basic) == target_repr


def test_multifrequency_pulse():
    members = [
        pulses.BasicPulse(0, 0.5, 1.5, 0.8, 40.00, 0.7, "Rectangular"),
        pulses.BasicPulse(1, 0.5, 5.0, 0.7, 100, 0.5, "Gaussian"),
        pulses.BasicPulse(2, 1.0, 3.5, 0.4, 70.0, 0.7, "Rectangular")
        ]
    multi = pulses.MultifrequencyPulse(members)
    target_repr = "M(P(0, 0.5, 1.5, 0.8, 40.0, 0.7, Rectangular), "\
                  "P(1, 0.5, 5.0, 0.7, 100, 0.5, Gaussian), "\
                  "P(2, 1.0, 3.5, 0.4, 70.0, 0.7, Rectangular))"
    assert repr(multi) == target_repr


def test_file_pulse():
    filep = pulses.FilePulse(0, 1.0, "testfile")
    target_repr = "F(0, 1.0, testfile)"
    assert repr(filep) == target_repr


def test_rectangular_shape():
    rect = pulses.Rectangular()
    assert rect.name == "rectangular"
    assert rect.envelope(1.0, 0.2, 2.2, 4.5) == 4.5


def test_gaussian_shape():
    gauss = pulses.Gaussian(1.5)
    assert gauss.name == "gaussian"
    assert gauss.sigma == 1.5
    assert repr(gauss) == "(gaussian, 1.5)"
    target_envelop = 4.4108940298803985
    time = np.array([1.0])
    assert gauss.envelope(time, 0.2, 2.2, 4.5) == target_envelop


def test_drag_shape():
    drag = pulses.Drag(1.5, 2.5)
    assert drag.name == "drag"
    assert drag.sigma == 1.5
    assert drag.beta == 2.5
    assert repr(drag) == "(drag, 1.5, 2.5)"
    target_envelop = 4.4108940298803985 + 1.470298009960133j
    time = np.array([1.0])
    assert drag.envelope(time, 0.2, 2.2, 4.5) == target_envelop


def test_swipht_shape():
    swipht = pulses.SWIPHT(2.2)
    assert swipht.name == "SWIPHT"
    assert swipht.g == 2.2
    assert repr(swipht) == "(SWIPHT, 2.2)"
    target_envelop = 4.4108940298803985
    time = np.array([1.0])
    assert swipht.envelope(time, 0.2, 2.2, 4.5) == 4.5


# TODO: Fix these tests so that waveform is not zero
def test_basic_pulse_compile():
    seq = PulseSequence([])
    waveform = np.zeros((seq.nchannels, seq.sample_size))
    basic = pulses.BasicPulse(0, 0.5, 1.5, 0.8, 40.00, 0.7, pulses.Gaussian(1.0))
    waveform = basic.compile(waveform, seq)
    target_waveform = np.zeros_like(waveform)
    np.testing.assert_allclose(waveform, target_waveform)


def test_multifrequency_pulse_compile():
    seq = PulseSequence([])
    waveform = np.zeros((seq.nchannels, seq.sample_size), dtype="complex128")
    members = [
        pulses.BasicPulse(0, 0.5, 1.5, 0.8, 40.00, 0.7, pulses.Gaussian(1.0)),
        pulses.BasicPulse(0, 0.5, 1.5, 0.8, 40.00, 0.7, pulses.Drag(1.0, 1.5))
        ]
    multi = pulses.MultifrequencyPulse(members)
    waveform = multi.compile(waveform, seq)
    target_waveform = np.zeros_like(waveform)
    np.testing.assert_allclose(waveform, target_waveform)


@pytest.mark.skip("Skipping this test because `sequence.file_dir` is not available")
def test_file_pulse_compile():
    seq = PulseSequence([])
    waveform = np.zeros((seq.nchannels, seq.sample_size))
    filep = pulses.FilePulse(0, 1.0, "file")
    waveform = filep.compile(waveform, seq)
    target_waveform = np.zeros_like(waveform)
    np.testing.assert_allclose(waveform, target_waveform)
