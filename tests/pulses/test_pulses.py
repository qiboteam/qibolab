"""Tests ``pulses.py``."""

import copy
import os
import pathlib

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
    plot,
)

HERE = pathlib.Path(__file__).parent


def test_plot_functions():
    p0 = Pulse(0, 40, 0.9, 0, 0, Rectangular(), 0, PulseType.FLUX, 0)
    p1 = Pulse(0, 40, 0.9, 50e6, 0, Gaussian(5), 0, PulseType.DRIVE, 2)
    p2 = Pulse(0, 40, 0.9, 50e6, 0, Drag(5, 2), 0, PulseType.DRIVE, 200)
    p3 = Pulse.flux(
        0, 40, 0.9, IIR([-0.5, 2], [1], Rectangular()), channel=0, qubit=200
    )
    p4 = Pulse.flux(0, 40, 0.9, SNZ(t_idling=10), channel=0, qubit=200)
    p5 = Pulse(0, 40, 0.9, 400e6, 0, eCap(alpha=2), 0, PulseType.DRIVE)
    p6 = Pulse(0, 40, 0.9, 50e6, 0, GaussianSquare(5, 0.9), 0, PulseType.DRIVE, 2)
    ps = PulseSequence([p0, p1, p2, p3, p4, p5, p6])
    wf = p0.modulated_waveform_i(0)

    plot_file = HERE / "test_plot.png"

    plot.waveform(wf, plot_file)
    assert os.path.exists(plot_file)
    os.remove(plot_file)

    plot.pulse(p0, plot_file)
    assert os.path.exists(plot_file)
    os.remove(plot_file)

    plot.sequence(ps, plot_file)
    assert os.path.exists(plot_file)
    os.remove(plot_file)


def test_pulse_init():
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


def test_pulse_attributes():
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


@pytest.mark.parametrize(
    "shape", [Rectangular(), Gaussian(5), GaussianSquare(5, 0.9), Drag(5, 1)]
)
def test_pulseshape_sampling_rate(shape):
    pulse = Pulse(0, 40, 0.9, 100e6, 0, shape, 0, PulseType.DRIVE)
    assert len(pulse.envelope_waveform_i(sampling_rate=1)) == 40
    assert len(pulse.envelope_waveform_i(sampling_rate=100)) == 4000


def testhape_eval():
    shape = PulseShape.eval("Rectangular()")
    assert isinstance(shape, Rectangular)
    with pytest.raises(ValueError):
        shape = PulseShape.eval("Ciao()")


@pytest.mark.parametrize("rel_sigma,beta", [(5, 1), (5, -1), (3, -0.03), (4, 0.02)])
def test_drag_shape_eval(rel_sigma, beta):
    shape = PulseShape.eval(f"Drag({rel_sigma}, {beta})")
    assert isinstance(shape, Drag)
    assert shape.rel_sigma == rel_sigma
    assert shape.beta == beta


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


def test_pulseshape_drag_shape():
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


def test_pulse_hash():
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


def test_pulse_aliases():
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


def test_pulse_pulse_order():
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


def modulate(
    i: np.ndarray,
    q: np.ndarray,
    num_samples: int,
    frequency: int,
    phase: float,
    sampling_rate: float,
):  #  -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(num_samples) / sampling_rate
    cosalpha = np.cos(2 * np.pi * frequency * time + phase)
    sinalpha = np.sin(2 * np.pi * frequency * time + phase)
    mod_matrix = np.array([[cosalpha, -sinalpha], [sinalpha, cosalpha]]) / np.sqrt(2)
    result = []
    for n, t, ii, qq in zip(np.arange(num_samples), time, i, q):
        result.append(mod_matrix[:, :, n] @ np.array([ii, qq]))
    mod_signals = np.array(result)
    return mod_signals[:, 0], mod_signals[:, 1]


def test_pulseshape_rectangular():
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
    global_phase = (
        2 * np.pi * _if * pulse.start / 1e9
    )  # pulse start, duration and finish are in ns
    mod_i, mod_q = modulate(
        i, q, num_samples, _if, global_phase + pulse.relative_phase, sampling_rate
    )

    np.testing.assert_allclose(pulse.shape.envelope_waveform_i(sampling_rate), i)
    np.testing.assert_allclose(pulse.shape.envelope_waveform_q(sampling_rate), q)
    np.testing.assert_allclose(
        pulse.shape.modulated_waveform_i(_if, sampling_rate), mod_i
    )
    np.testing.assert_allclose(
        pulse.shape.modulated_waveform_q(_if, sampling_rate), mod_q
    )


def test_pulseshape_gaussian():
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
    global_phase = (
        2 * np.pi * pulse.frequency * pulse.start / 1e9
    )  # pulse start, duration and finish are in ns
    mod_i, mod_q = modulate(
        i, q, num_samples, _if, global_phase + pulse.relative_phase, sampling_rate
    )

    np.testing.assert_allclose(pulse.shape.envelope_waveform_i(sampling_rate), i)
    np.testing.assert_allclose(pulse.shape.envelope_waveform_q(sampling_rate), q)
    np.testing.assert_allclose(
        pulse.shape.modulated_waveform_i(_if, sampling_rate), mod_i
    )
    np.testing.assert_allclose(
        pulse.shape.modulated_waveform_q(_if, sampling_rate), mod_q
    )


def test_pulseshape_drag():
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
    global_phase = (
        2 * np.pi * _if * pulse.start / 1e9
    )  # pulse start, duration and finish are in ns
    mod_i, mod_q = modulate(
        i, q, num_samples, _if, global_phase + pulse.relative_phase, sampling_rate
    )

    np.testing.assert_allclose(pulse.shape.envelope_waveform_i(sampling_rate), i)
    np.testing.assert_allclose(pulse.shape.envelope_waveform_q(sampling_rate), q)
    np.testing.assert_allclose(
        pulse.shape.modulated_waveform_i(_if, sampling_rate), mod_i
    )
    np.testing.assert_allclose(
        pulse.shape.modulated_waveform_q(_if, sampling_rate), mod_q
    )


def test_pulseshape_eq():
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
