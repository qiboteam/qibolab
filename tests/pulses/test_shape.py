import numpy as np
import pytest

from qibolab.pulses import (
    IIR,
    SNZ,
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
from qibolab.pulses.shape import demodulate, modulate


@pytest.mark.parametrize(
    "shape", [Rectangular(), Gaussian(5), GaussianSquare(5, 0.9), Drag(5, 1)]
)
def test_sampling_rate(shape):
    pulse = Pulse(0, 40, 0.9, 100e6, 0, shape, 0, PulseType.DRIVE)
    assert len(pulse.envelope_waveform_i(sampling_rate=1)) == 40
    assert len(pulse.envelope_waveform_i(sampling_rate=100)) == 4000


def test_eval():
    shape = PulseShape.eval("Rectangular()")
    assert isinstance(shape, Rectangular)
    shape = PulseShape.eval("Drag(5, 1)")
    assert isinstance(shape, Drag)
    with pytest.raises(ValueError):
        shape = PulseShape.eval("Ciao()")


def test_raise_shapeiniterror():
    shape = Rectangular()
    with pytest.raises(ShapeInitError):
        shape.envelope_waveform_i()
    with pytest.raises(ShapeInitError):
        shape.envelope_waveform_q()

    shape = Gaussian(0)
    with pytest.raises(ShapeInitError):
        shape.envelope_waveform_i()
    with pytest.raises(ShapeInitError):
        shape.envelope_waveform_q()

    shape = GaussianSquare(0, 1)
    with pytest.raises(ShapeInitError):
        shape.envelope_waveform_i()
    with pytest.raises(ShapeInitError):
        shape.envelope_waveform_q()

    shape = Drag(0, 0)
    with pytest.raises(ShapeInitError):
        shape.envelope_waveform_i()
    with pytest.raises(ShapeInitError):
        shape.envelope_waveform_q()

    shape = IIR([0], [0], None)
    with pytest.raises(ShapeInitError):
        shape.envelope_waveform_i()
    with pytest.raises(ShapeInitError):
        shape.envelope_waveform_q()

    shape = SNZ(0)
    with pytest.raises(ShapeInitError):
        shape.envelope_waveform_i()
    with pytest.raises(ShapeInitError):
        shape.envelope_waveform_q()

    shape = eCap(0)
    with pytest.raises(ShapeInitError):
        shape.envelope_waveform_i()
    with pytest.raises(ShapeInitError):
        shape.envelope_waveform_q()


def test_drag_shape():
    pulse = Pulse(0, 2, 1, 4e9, 0, Drag(2, 1), 0, PulseType.DRIVE)
    # envelope i & envelope q should cross nearly at 0 and at 2
    waveform = pulse.envelope_waveform_i(sampling_rate=10)
    target_waveform = np.array(
        [
            0.63683161,
            0.69680478,
            0.7548396,
            0.80957165,
            0.85963276,
            0.90370708,
            0.94058806,
            0.96923323,
            0.98881304,
            0.99875078,
            0.99875078,
            0.98881304,
            0.96923323,
            0.94058806,
            0.90370708,
            0.85963276,
            0.80957165,
            0.7548396,
            0.69680478,
            0.63683161,
        ]
    )
    np.testing.assert_allclose(waveform, target_waveform)


def test_rectangular():
    pulse = Pulse(
        start=0,
        duration=50,
        amplitude=1,
        frequency=200_000_000,
        relative_phase=0,
        shape=Rectangular(),
        channel=1,
        qubit=0,
    )
    _if = 0

    assert pulse.duration == 50
    assert isinstance(pulse.shape, Rectangular)
    assert pulse.shape.name == "Rectangular"
    assert repr(pulse.shape) == "Rectangular()"

    sampling_rate = 1
    num_samples = int(pulse.duration / sampling_rate)
    i, q = (
        pulse.amplitude * np.ones(num_samples),
        pulse.amplitude * np.zeros(num_samples),
    )

    np.testing.assert_allclose(pulse.shape.envelope_waveform_i(sampling_rate), i)
    np.testing.assert_allclose(pulse.shape.envelope_waveform_q(sampling_rate), q)


def test_gaussian():
    pulse = Pulse(
        start=0,
        duration=50,
        amplitude=1,
        frequency=200_000_000,
        relative_phase=0,
        shape=Gaussian(5),
        channel=1,
        qubit=0,
    )
    _if = 0

    assert pulse.duration == 50
    assert isinstance(pulse.shape, Gaussian)
    assert pulse.shape.name == "Gaussian"
    assert pulse.shape.rel_sigma == 5
    assert repr(pulse.shape) == "Gaussian(5)"

    sampling_rate = 1
    num_samples = int(pulse.duration / sampling_rate)
    x = np.arange(0, num_samples, 1)
    i = pulse.amplitude * np.exp(
        -(1 / 2)
        * (
            ((x - (num_samples - 1) / 2) ** 2)
            / (((num_samples) / pulse.shape.rel_sigma) ** 2)
        )
    )
    q = pulse.amplitude * np.zeros(num_samples)

    np.testing.assert_allclose(pulse.shape.envelope_waveform_i(sampling_rate), i)
    np.testing.assert_allclose(pulse.shape.envelope_waveform_q(sampling_rate), q)


def test_drag():
    pulse = Pulse(
        start=0,
        duration=50,
        amplitude=1,
        frequency=200_000_000,
        relative_phase=0,
        shape=Drag(5, 0.2),
        channel=1,
        qubit=0,
    )
    _if = 0

    assert pulse.duration == 50
    assert isinstance(pulse.shape, Drag)
    assert pulse.shape.name == "Drag"
    assert pulse.shape.rel_sigma == 5
    assert pulse.shape.beta == 0.2
    assert repr(pulse.shape) == "Drag(5, 0.2)"

    sampling_rate = 1
    num_samples = int(pulse.duration / 1 * sampling_rate)
    x = np.arange(0, num_samples, 1)
    i = pulse.amplitude * np.exp(
        -(1 / 2)
        * (
            ((x - (num_samples - 1) / 2) ** 2)
            / (((num_samples) / pulse.shape.rel_sigma) ** 2)
        )
    )
    q = (
        pulse.shape.beta
        * (-(x - (num_samples - 1) / 2) / ((num_samples / pulse.shape.rel_sigma) ** 2))
        * i
        * sampling_rate
    )

    np.testing.assert_allclose(pulse.shape.envelope_waveform_i(sampling_rate), i)
    np.testing.assert_allclose(pulse.shape.envelope_waveform_q(sampling_rate), q)


def test_eq():
    """Checks == operator for pulse shapes."""

    shape1 = Rectangular()
    shape2 = Rectangular()
    shape3 = Gaussian(5)
    assert shape1 == shape2
    assert not shape1 == shape3

    shape1 = Gaussian(4)
    shape2 = Gaussian(4)
    shape3 = Gaussian(5)
    assert shape1 == shape2
    assert not shape1 == shape3

    shape1 = GaussianSquare(4, 0.01)
    shape2 = GaussianSquare(4, 0.01)
    shape3 = GaussianSquare(5, 0.01)
    shape4 = GaussianSquare(4, 0.05)
    shape5 = GaussianSquare(5, 0.05)
    assert shape1 == shape2
    assert not shape1 == shape3
    assert not shape1 == shape4
    assert not shape1 == shape5

    shape1 = Drag(4, 0.01)
    shape2 = Drag(4, 0.01)
    shape3 = Drag(5, 0.01)
    shape4 = Drag(4, 0.05)
    shape5 = Drag(5, 0.05)
    assert shape1 == shape2
    assert not shape1 == shape3
    assert not shape1 == shape4
    assert not shape1 == shape5

    shape1 = IIR([-0.5, 2], [1], Rectangular())
    shape2 = IIR([-0.5, 2], [1], Rectangular())
    shape3 = IIR([-0.5, 4], [1], Rectangular())
    shape4 = IIR([-0.4, 2], [1], Rectangular())
    shape5 = IIR([-0.5, 2], [2], Rectangular())
    shape6 = IIR([-0.5, 2], [2], Gaussian(5))
    assert shape1 == shape2
    assert not shape1 == shape3
    assert not shape1 == shape4
    assert not shape1 == shape5
    assert not shape1 == shape6

    shape1 = SNZ(5)
    shape2 = SNZ(5)
    shape3 = SNZ(2)
    shape4 = SNZ(2, 0.1)
    shape5 = SNZ(2, 0.1)
    assert shape1 == shape2
    assert not shape1 == shape3
    assert not shape1 == shape4
    assert not shape1 == shape5

    shape1 = eCap(4)
    shape2 = eCap(4)
    shape3 = eCap(5)
    assert shape1 == shape2
    assert not shape1 == shape3


def test_demodulation():
    signal = np.ones((2, 100))
    freq = 0.15
    mod = modulate(signal, freq)

    demod = demodulate(mod, freq)
    np.testing.assert_allclose(demod, signal)

    mod1 = modulate(demod, freq * 3.0, rate=3.0)
    np.testing.assert_allclose(mod1, mod)

    mod2 = modulate(signal, freq, phase=2 * np.pi)
    np.testing.assert_allclose(mod2, mod)

    demod1 = demodulate(mod + np.ones_like(mod), freq)
    np.testing.assert_allclose(demod1, demod)
