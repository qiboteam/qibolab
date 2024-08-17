import numpy as np
import pytest

from qibolab.pulses import (
    Drag,
    ECap,
    Gaussian,
    GaussianSquare,
    Iir,
    Pulse,
    Rectangular,
    Snz,
)


@pytest.mark.parametrize(
    "shape",
    [
        Rectangular(),
        Gaussian(rel_sigma=5),
        GaussianSquare(rel_sigma=5, width=0.9),
        Drag(rel_sigma=5, beta=1),
    ],
)
def test_sampling_rate(shape):
    pulse = Pulse(
        duration=40,
        amplitude=0.9,
        envelope=shape,
        relative_phase=0,
    )
    assert len(pulse.i(sampling_rate=1)) == 40
    assert len(pulse.i(sampling_rate=100)) == 4000


def test_drag_shape():
    pulse = Pulse(
        duration=2,
        amplitude=1,
        envelope=Drag(rel_sigma=0.5, beta=1),
        relative_phase=0,
    )
    # envelope i & envelope q should cross nearly at 0 and at 2
    waveform = pulse.i(sampling_rate=10)
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
        duration=50,
        amplitude=1,
        relative_phase=0,
        envelope=Rectangular(),
    )

    assert pulse.duration == 50
    assert isinstance(pulse.envelope, Rectangular)

    sampling_rate = 1
    num_samples = int(pulse.duration / sampling_rate)
    i, q = (
        pulse.amplitude * np.ones(num_samples),
        pulse.amplitude * np.zeros(num_samples),
    )

    np.testing.assert_allclose(pulse.i(sampling_rate), i)
    np.testing.assert_allclose(pulse.q(sampling_rate), q)


def test_gaussian():
    pulse = Pulse(
        duration=50,
        amplitude=1,
        relative_phase=0,
        envelope=Gaussian(rel_sigma=5),
    )

    assert pulse.duration == 50
    assert isinstance(pulse.envelope, Gaussian)
    assert pulse.envelope.rel_sigma == 5

    sampling_rate = 1
    num_samples = int(pulse.duration / sampling_rate)
    x = np.arange(0, num_samples, 1)
    i = pulse.amplitude * np.exp(
        -(
            ((x - (num_samples - 1) / 2) ** 2)
            / (2 * (num_samples * pulse.envelope.rel_sigma) ** 2)
        )
    )
    q = pulse.amplitude * np.zeros(num_samples)

    np.testing.assert_allclose(pulse.i(sampling_rate), i)
    np.testing.assert_allclose(pulse.q(sampling_rate), q)


def test_drag():
    pulse = Pulse(
        duration=50,
        amplitude=1,
        relative_phase=0,
        envelope=Drag(rel_sigma=0.2, beta=0.2),
    )

    assert pulse.duration == 50
    assert isinstance(pulse.envelope, Drag)
    assert pulse.envelope.rel_sigma == 0.2
    assert pulse.envelope.beta == 0.2

    sampling_rate = 1
    num_samples = int(pulse.duration / sampling_rate)
    x = np.arange(num_samples)
    i = pulse.amplitude * np.exp(
        -(
            ((x - (num_samples - 1) / 2) ** 2)
            / (2 * (num_samples * pulse.envelope.rel_sigma) ** 2)
        )
    )
    q = pulse.amplitude * (
        pulse.envelope.beta
        * (
            -(x - (num_samples - 1) / 2)
            / ((num_samples * pulse.envelope.rel_sigma) ** 2)
        )
        * i
        * sampling_rate
    )

    np.testing.assert_allclose(pulse.i(sampling_rate), i)
    np.testing.assert_allclose(pulse.q(sampling_rate), q)


def test_eq():
    """Checks == operator for pulse shapes."""

    shape1 = Rectangular()
    shape2 = Rectangular()
    shape3 = Gaussian(rel_sigma=5)
    assert shape1 == shape2
    assert not shape1 == shape3

    shape1 = Gaussian(rel_sigma=4)
    shape2 = Gaussian(rel_sigma=4)
    shape3 = Gaussian(rel_sigma=5)
    assert shape1 == shape2
    assert not shape1 == shape3

    shape1 = GaussianSquare(rel_sigma=4, width=0.01)
    shape2 = GaussianSquare(rel_sigma=4, width=0.01)
    shape3 = GaussianSquare(rel_sigma=5, width=0.01)
    shape4 = GaussianSquare(rel_sigma=4, width=0.05)
    shape5 = GaussianSquare(rel_sigma=5, width=0.05)
    assert shape1 == shape2
    assert not shape1 == shape3
    assert not shape1 == shape4
    assert not shape1 == shape5

    shape1 = Drag(rel_sigma=4, beta=0.01)
    shape2 = Drag(rel_sigma=4, beta=0.01)
    shape3 = Drag(rel_sigma=5, beta=0.01)
    shape4 = Drag(rel_sigma=4, beta=0.05)
    shape5 = Drag(rel_sigma=5, beta=0.05)
    assert shape1 == shape2
    assert not shape1 == shape3
    assert not shape1 == shape4
    assert not shape1 == shape5

    shape1 = Iir(a=np.array([-0.5, 2]), b=np.array([1]), target=Rectangular())
    shape2 = Iir(a=np.array([-0.5, 2]), b=np.array([1]), target=Rectangular())
    shape3 = Iir(a=np.array([-0.5, 4]), b=np.array([1]), target=Rectangular())
    shape4 = Iir(a=np.array([-0.4, 2]), b=np.array([1]), target=Rectangular())
    shape5 = Iir(a=np.array([-0.5, 2]), b=np.array([2]), target=Rectangular())
    shape6 = Iir(a=np.array([-0.5, 2]), b=np.array([2]), target=Gaussian(rel_sigma=5))
    assert shape1 == shape2
    assert not shape1 == shape3
    assert not shape1 == shape4
    assert not shape1 == shape5
    assert not shape1 == shape6

    shape1 = Snz(t_idling=5)
    shape2 = Snz(t_idling=5)
    shape3 = Snz(t_idling=2)
    shape4 = Snz(t_idling=2, b_amplitude=0.1)
    shape5 = Snz(t_idling=2, b_amplitude=0.1)
    assert shape1 == shape2
    assert not shape1 == shape3
    assert not shape1 == shape4
    assert not shape1 == shape5

    shape1 = ECap(alpha=4)
    shape2 = ECap(alpha=4)
    shape3 = ECap(alpha=5)
    assert shape1 == shape2
    assert not shape1 == shape3
