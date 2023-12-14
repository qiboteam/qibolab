"""Tests ``pulses.py``."""
import os
import pathlib

import numpy as np
import pytest

from qibolab.pulses import (
    IIR,
    SNZ,
    Custom,
    Drag,
    DrivePulse,
    FluxPulse,
    Gaussian,
    Pulse,
    PulseSequence,
    PulseShape,
    PulseType,
    ReadoutPulse,
    Rectangular,
    ShapeInitError,
    SplitPulse,
    Waveform,
    eCap,
)
from qibolab.symbolic import intSymbolicExpression as se_int

HERE = pathlib.Path(__file__).parent


def test_pulses_plot_functions():
    p0 = Pulse(0, 40, 0.9, 0, 0, Rectangular(), 0, PulseType.FLUX, 0)
    p1 = Pulse(0, 40, 0.9, 50e6, 0, Gaussian(5), 0, PulseType.DRIVE, 2)
    p2 = Pulse(0, 40, 0.9, 50e6, 0, Drag(5, 2), 0, PulseType.DRIVE, 200)
    p3 = FluxPulse(0, 40, 0.9, IIR([-0.5, 2], [1], Rectangular()), 0, 200)
    p4 = FluxPulse(0, 40, 0.9, SNZ(t_idling=10), 0, 200)
    p5 = Pulse(0, 40, 0.9, 400e6, 0, eCap(alpha=2), 0, PulseType.DRIVE)
    p6 = SplitPulse(p5, window_start=10, window_finish=30)
    ps = p0 + p1 + p2 + p3 + p4 + p5 + p6
    wf = p0.modulated_waveform_i()

    plot_file = HERE / "test_plot.png"

    wf.plot(plot_file)
    assert os.path.exists(plot_file)
    os.remove(plot_file)

    p0.plot(plot_file)
    assert os.path.exists(plot_file)
    os.remove(plot_file)

    p6.plot(plot_file)
    assert os.path.exists(plot_file)
    os.remove(plot_file)

    ps.plot(plot_file)
    assert os.path.exists(plot_file)
    os.remove(plot_file)


def test_pulses_pulse_init():
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
    assert repr(p0) == "Pulse(0, 50, 0.9, 20_000_000, 0, Rectangular(), 0, PulseType.READOUT, 0)"

    # initialisation with Symbolic Expressions
    t1 = se_int(100, "t1")
    d1 = se_int(50, "d1")
    p1 = Pulse(
        start=t1,
        duration=d1,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=0.0,
        shape=Rectangular(),
        channel=0,
        type=PulseType.READOUT,
        qubit=0,
    )
    assert repr(p1) == "Pulse(100, 50, 0.9, 20_000_000, 0, Rectangular(), 0, PulseType.READOUT, 0)"

    # initialisation with non int (float) frequency
    p2 = Pulse(
        start=0,
        duration=50,
        amplitude=0.9,
        frequency=20e6,
        relative_phase=0,
        shape=Rectangular(),
        channel=0,
        type=PulseType.READOUT,
        qubit=0,
    )
    assert repr(p2) == "Pulse(0, 50, 0.9, 20_000_000, 0, Rectangular(), 0, PulseType.READOUT, 0)"
    assert type(p2.frequency) == int and p2.frequency == 20_000_000

    # initialisation with non float (int) relative_phase
    p3 = Pulse(
        start=0,
        duration=50,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=1,
        shape=Rectangular(),
        channel=0,
        type=PulseType.READOUT,
        qubit=0,
    )
    assert repr(p3) == "Pulse(0, 50, 0.9, 20_000_000, 1, Rectangular(), 0, PulseType.READOUT, 0)"
    assert type(p3.relative_phase) == float and p3.relative_phase == 1.0

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
    assert repr(p4) == "Pulse(0, 50, 0.9, 20_000_000, 0, Rectangular(), 0, PulseType.READOUT, 0)"

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
    assert repr(p5) == "Pulse(0, 50, 0.9, 20_000_000, 0, Rectangular(), channel0, PulseType.READOUT, qubit0)"
    assert p5.qubit == "qubit0"

    # initialisation with different frequencies, shapes and types
    p6 = Pulse(0, 40, 0.9, -50e6, 0, Rectangular(), 0, PulseType.READOUT)
    p7 = Pulse(0, 40, 0.9, 0, 0, Rectangular(), 0, PulseType.FLUX, 0)
    p8 = Pulse(0, 40, 0.9, 50e6, 0, Gaussian(5), 0, PulseType.DRIVE, 2)
    p9 = Pulse(0, 40, 0.9, 50e6, 0, Drag(5, 2), 0, PulseType.DRIVE, 200)
    p10 = FluxPulse(0, 40, 0.9, IIR([-1, 1], [-0.1, 0.1001], Rectangular()), 0, 200)
    p11 = FluxPulse(0, 40, 0.9, SNZ(t_idling=10, b_amplitude=0.5), 0, 200)
    p11 = Pulse(0, 40, 0.9, 400e6, 0, eCap(alpha=2), 0, PulseType.DRIVE)

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
    assert repr(p12) == "Pulse(5.5, 34.33, 0.9, 20_000_000, 1, Rectangular(), 0, PulseType.READOUT, 0)"
    assert isinstance(p12.start, float)
    assert isinstance(p12.duration, float)
    assert p12.finish == 5.5 + 34.33


def test_pulses_pulse_attributes():
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
        type=PulseType.READOUT,
        qubit=qubit,
    )

    assert type(p10.start) == int and p10.start == 10
    assert type(p10.duration) == int and p10.duration == 50
    assert type(p10.amplitude) == float and p10.amplitude == 0.9
    assert type(p10.frequency) == int and p10.frequency == 20_000_000
    assert type(p10.phase) == float and np.allclose(p10.phase, 2 * np.pi * p10.start * p10.frequency / 1e9)
    assert isinstance(p10.shape, PulseShape) and repr(p10.shape) == "Rectangular()"
    assert type(p10.channel) == type(channel) and p10.channel == channel
    assert type(p10.qubit) == type(qubit) and p10.qubit == qubit
    assert type(p10.finish) == int and p10.finish == 60

    ValueError_raised = False
    try:
        p10 = Pulse(
            start=-10,  # Start should be >= 0
            duration=50,
            amplitude=0.9,
            frequency=20_000_000,
            relative_phase=0.0,
            shape=Rectangular(),
            channel=channel,
            type=PulseType.READOUT,
            qubit=qubit,
        )
    except ValueError:
        ValueError_raised = True
    except:
        assert False
    assert ValueError_raised

    ValueError_raised = False
    try:
        p10 = Pulse(
            start=0,
            duration=-1,  # duration should be > 0
            amplitude=0.9,
            frequency=20_000_000,
            relative_phase=0.0,
            shape=Rectangular(),
            channel=channel,
            type=PulseType.READOUT,
            qubit=qubit,
        )
    except ValueError:
        ValueError_raised = True
    except:
        assert False
    assert ValueError_raised

    ValueError_raised = False
    try:
        p10 = Pulse(
            start=0,
            duration=50,
            amplitude=1.1,  # amplitude should be >= 0 & <= 1
            frequency=20_000_000,
            relative_phase=0.0,
            shape=Rectangular(),
            channel=channel,
            type=PulseType.READOUT,
            qubit=qubit,
        )
    except ValueError:
        ValueError_raised = True
    except:
        assert False
    assert ValueError_raised

    ValueError_raised = False
    try:
        p10 = Pulse(
            start=0,
            duration=50,
            amplitude=0.9,
            frequency=20_000_000,
            relative_phase=0.0,
            shape="NonImplementedShape()",
            channel=channel,
            type=PulseType.READOUT,
            qubit=qubit,
        )
    except ValueError:
        ValueError_raised = True
    except:
        assert False
    assert ValueError_raised

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


def test_pulses_is_equal_ignoring_start():
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


def test_pulses_pulse_serial():
    p11 = Pulse(0, 40, 0.9, 50_000_000, 0, Gaussian(5), 0, PulseType.DRIVE)
    assert p11.serial == "Pulse(0, 40, 0.9, 50_000_000, 0, Gaussian(5), 0, PulseType.DRIVE, 0)"
    assert repr(p11) == p11.serial


@pytest.mark.parametrize("shape", [Rectangular(), Gaussian(5), Drag(5, 1)])
def test_pulses_pulseshape_sampling_rate(shape):
    pulse = Pulse(0, 40, 0.9, 100e6, 0, shape, 0, PulseType.DRIVE)
    assert len(pulse.envelope_waveform_i(sampling_rate=1).data) == 40
    assert len(pulse.envelope_waveform_i(sampling_rate=100).data) == 4000


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


def test_pulses_pulseshape_drag_shape():
    pulse = Pulse(0, 2, 1, 4e9, 0, Drag(2, 1), 0, PulseType.DRIVE)
    # envelope i & envelope q should cross nearly at 0 and at 2
    waveform = pulse.envelope_waveform_i(sampling_rate=10).data
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


def test_pulses_pulse_hash():
    rp = Pulse(0, 40, 0.9, 100e6, 0, Rectangular(), 0, PulseType.DRIVE)
    dp = Pulse(0, 40, 0.9, 100e6, 0, Drag(5, 1), 0, PulseType.DRIVE)
    hash(rp)
    my_dict = {rp: 1, dp: 2}
    assert list(my_dict.keys())[0] == rp
    assert list(my_dict.keys())[1] == dp

    p1 = Pulse(0, 40, 0.9, 100e6, 0, Drag(5, 1), 0, PulseType.DRIVE)
    p2 = Pulse(0, 40, 0.9, 100e6, 0, Drag(5, 1), 0, PulseType.DRIVE)

    assert p1 == p2

    t0 = se_int(0, "t0")
    t1 = se_int(0, "t1")
    p1 = Pulse(t0, 40, 0.9, 100e6, 0, Drag(5, 1), 0, PulseType.DRIVE)
    p2 = Pulse(t1, 40, 0.9, 100e6, 0, Drag(5, 1), 0, PulseType.DRIVE)
    assert p1 == p2
    t0 += 100
    assert p1 != p2

    t0 = se_int(0, "t0")
    p1 = Pulse(t0, 40, 0.9, 100e6, 0, Drag(5, 1), 0, PulseType.DRIVE)
    p2 = p1.shallow_copy()
    p3 = p1.copy()
    assert p1 == p2
    assert p1 == p3

    t0 += 100
    assert p1 == p2
    assert p1 != p3


def test_pulses_pulse_aliases():
    rop = ReadoutPulse(
        start=0,
        duration=50,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=0.0,
        shape=Rectangular(),
        channel=0,
        qubit=0,
    )
    assert repr(rop) == "ReadoutPulse(0, 50, 0.9, 20_000_000, 0, Rectangular(), 0, 0)"

    dp = DrivePulse(
        start=0,
        duration=2000,
        amplitude=0.9,
        frequency=200_000_000,
        relative_phase=0.0,
        shape=Gaussian(5),
        channel=0,
        qubit=0,
    )
    assert repr(dp) == "DrivePulse(0, 2000, 0.9, 200_000_000, 0, Gaussian(5), 0, 0)"

    fp = FluxPulse(start=0, duration=300, amplitude=0.9, shape=Rectangular(), channel=0, qubit=0)
    assert repr(fp) == "FluxPulse(0, 300, 0.9, Rectangular(), 0, 0)"


def test_pulses_pulse_split_pulse():
    dp = Pulse(
        start=500,
        duration=2000,
        amplitude=0.9,
        frequency=5_000_000,
        relative_phase=0.0,
        shape=Rectangular(),
        channel=0,
        type=PulseType.READOUT,
        qubit=0,
    )

    sp = SplitPulse(dp)
    sp.channel = 1
    a = 720
    b = 960
    sp.window_start = sp.start + a
    sp.window_finish = sp.start + b
    assert sp.window_start == sp.start + a
    assert sp.window_finish == sp.start + b
    ps = PulseSequence(dp, sp)
    # ps.plot()
    assert len(sp.envelope_waveform_i()) == b - a
    assert len(sp.envelope_waveform_q()) == b - a
    assert len(sp.modulated_waveform_i()) == b - a
    assert len(sp.modulated_waveform_q()) == b - a


def test_pulses_pulsesequence_init():
    p1 = Pulse(400, 40, 0.9, 100e6, 0, Drag(5, 1), 3, PulseType.DRIVE)
    p2 = Pulse(500, 40, 0.9, 100e6, 0, Drag(5, 1), 2, PulseType.DRIVE)
    p3 = Pulse(600, 40, 0.9, 100e6, 0, Drag(5, 1), 1, PulseType.DRIVE)

    ps = PulseSequence()
    assert type(ps) == PulseSequence

    ps = PulseSequence(p1, p2, p3)
    assert ps.count == 3 and len(ps) == 3
    assert ps[0] == p1
    assert ps[1] == p2
    assert ps[2] == p3

    other_ps = p1 + p2 + p3
    assert other_ps.count == 3 and len(other_ps) == 3
    assert other_ps[0] == p1
    assert other_ps[1] == p2
    assert other_ps[2] == p3

    plist = [p1, p2, p3]
    n = 0
    for pulse in ps:
        assert plist[n] == pulse
        n += 1


def test_pulses_pulsesequence_operators():
    ps = PulseSequence()
    ps += ReadoutPulse(800, 200, 0.9, 20e6, 0, Rectangular(), 1)
    ps = ps + ReadoutPulse(800, 200, 0.9, 20e6, 0, Rectangular(), 2)
    ps = ReadoutPulse(800, 200, 0.9, 20e6, 0, Rectangular(), 3) + ps

    p4 = Pulse(100, 40, 0.9, 50e6, 0, Gaussian(5), 3, PulseType.DRIVE)
    p5 = Pulse(200, 40, 0.9, 50e6, 0, Gaussian(5), 2, PulseType.DRIVE)
    p6 = Pulse(300, 40, 0.9, 50e6, 0, Gaussian(5), 1, PulseType.DRIVE)

    another_ps = PulseSequence()
    another_ps.add(p4)
    another_ps.add(p5, p6)

    assert another_ps[0] == p4
    assert another_ps[1] == p5
    assert another_ps[2] == p6

    ps += another_ps

    assert ps.count == 6
    assert p5 in ps

    # ps.plot()

    p7 = Pulse(600, 40, 0.9, 100e6, 0, Drag(5, 1), 1, PulseType.DRIVE)
    yet_another_ps = PulseSequence(p7)
    assert yet_another_ps.count == 1
    yet_another_ps *= 3
    assert yet_another_ps.count == 3
    yet_another_ps *= 3
    assert yet_another_ps.count == 9

    p8 = Pulse(600, 40, 0.9, 100e6, 0, Drag(5, 1), 1, PulseType.DRIVE)
    p9 = Pulse(600, 40, 0.9, 100e6, 0, Drag(5, 1), 2, PulseType.DRIVE)
    and_yet_another_ps = 2 * p9 + p8 * 3
    assert and_yet_another_ps.count == 5


def test_pulses_pulsesequence_add():
    p0 = Pulse(0, 40, 0.9, 50e6, 0, Gaussian(5), 10, PulseType.DRIVE, 1)
    p1 = Pulse(100, 40, 0.9, 50e6, 0, Gaussian(5), 20, PulseType.DRIVE, 2)

    p2 = Pulse(200, 40, 0.9, 50e6, 0, Gaussian(5), 30, PulseType.DRIVE, 3)
    p3 = Pulse(400, 40, 0.9, 50e6, 0, Gaussian(5), 40, PulseType.DRIVE, 4)

    ps = PulseSequence()
    ps.add(p0)
    ps.add(p1)
    psx = PulseSequence(p2, p3)
    ps.add(psx)

    assert ps.count == 4
    assert ps.qubits == [1, 2, 3, 4]
    assert ps.channels == [10, 20, 30, 40]
    assert ps.start == 0
    assert ps.finish == 440


def test_pulses_pulsesequence_clear():
    p1 = Pulse(600, 40, 0.9, 100e6, 0, Drag(5, 1), 1, PulseType.DRIVE)
    p2 = Pulse(600, 40, 0.9, 100e6, 0, Drag(5, 1), 2, PulseType.DRIVE)
    ps = 2 * p2 + p1 * 3
    assert ps.count == 5
    ps.clear()
    assert ps.count == 0
    assert ps.is_empty


def test_pulses_pulsesequence_start_finish():
    p1 = Pulse(20, 40, 0.9, 200e6, 0, Drag(5, 1), 1, PulseType.DRIVE)
    p2 = Pulse(60, 1000, 0.9, 20e6, 0, Rectangular(), 2, PulseType.READOUT)
    ps = p1 + p2
    assert ps.start == p1.start
    assert ps.finish == p2.finish


def test_pulses_pulsesequence_get_channel_pulses():
    p1 = DrivePulse(0, 400, 0.9, 20e6, 0, Gaussian(5), 10)
    p2 = ReadoutPulse(100, 400, 0.9, 20e6, 0, Rectangular(), 30)
    p3 = DrivePulse(300, 400, 0.9, 20e6, 0, Drag(5, 50), 20)
    p4 = DrivePulse(400, 400, 0.9, 20e6, 0, Drag(5, 50), 30)
    p5 = ReadoutPulse(500, 400, 0.9, 20e6, 0, Rectangular(), 20)
    p6 = DrivePulse(600, 400, 0.9, 20e6, 0, Gaussian(5), 30)

    ps = PulseSequence(p1, p2, p3, p4, p5, p6)
    assert ps.channels == [10, 20, 30]
    assert ps.get_channel_pulses(10).count == 1
    assert ps.get_channel_pulses(20).count == 2
    assert ps.get_channel_pulses(30).count == 3
    assert ps.get_channel_pulses(20, 30).count == 5


def test_pulses_pulsesequence_get_qubit_pulses():
    p1 = DrivePulse(0, 400, 0.9, 20e6, 0, Gaussian(5), 10, 0)
    p2 = ReadoutPulse(100, 400, 0.9, 20e6, 0, Rectangular(), 30, 0)
    p3 = DrivePulse(300, 400, 0.9, 20e6, 0, Drag(5, 50), 20, 1)
    p4 = DrivePulse(400, 400, 0.9, 20e6, 0, Drag(5, 50), 30, 1)
    p5 = ReadoutPulse(500, 400, 0.9, 20e6, 0, Rectangular(), 30, 1)
    p6 = FluxPulse(600, 400, 0.9, Rectangular(), 40, 1)
    p7 = FluxPulse(900, 400, 0.9, Rectangular(), 40, 2)

    ps = PulseSequence(p1, p2, p3, p4, p5, p6, p7)
    assert ps.qubits == [0, 1, 2]
    assert ps.get_qubit_pulses(0).count == 2
    assert ps.get_qubit_pulses(1).count == 4
    assert ps.get_qubit_pulses(2).count == 1
    assert ps.get_qubit_pulses(0, 1).count == 6


def test_pulses_pulsesequence_pulses_overlap():
    p1 = DrivePulse(0, 400, 0.9, 20e6, 0, Gaussian(5), 10)
    p2 = ReadoutPulse(100, 400, 0.9, 20e6, 0, Rectangular(), 30)
    p3 = DrivePulse(300, 400, 0.9, 20e6, 0, Drag(5, 50), 20)
    p4 = DrivePulse(400, 400, 0.9, 20e6, 0, Drag(5, 50), 30)
    p5 = ReadoutPulse(500, 400, 0.9, 20e6, 0, Rectangular(), 20)
    p6 = DrivePulse(600, 400, 0.9, 20e6, 0, Gaussian(5), 30)

    ps = PulseSequence(p1, p2, p3, p4, p5, p6)
    assert ps.pulses_overlap == True
    assert ps.get_channel_pulses(10).pulses_overlap == False
    assert ps.get_channel_pulses(20).pulses_overlap == True
    assert ps.get_channel_pulses(30).pulses_overlap == True

    channel10_ps = ps.get_channel_pulses(10)
    channel20_ps = ps.get_channel_pulses(20)
    channel30_ps = ps.get_channel_pulses(30)

    split_pulses = PulseSequence()
    overlaps = channel20_ps.get_pulse_overlaps()
    n = 0
    for section in overlaps.keys():
        for pulse in overlaps[section]:
            sp = SplitPulse(pulse, section[0], section[1])
            sp.channel = n
            split_pulses.add(sp)
            n += 1
    # split_pulses.plot()


def test_pulses_pulsesequence_separate_overlapping_pulses():
    p1 = DrivePulse(0, 400, 0.9, 20e6, 0, Gaussian(5), 10)
    p2 = ReadoutPulse(100, 400, 0.9, 20e6, 0, Rectangular(), 30)
    p3 = DrivePulse(300, 400, 0.9, 20e6, 0, Drag(5, 50), 20)
    p4 = DrivePulse(400, 400, 0.9, 20e6, 0, Drag(5, 50), 30)
    p5 = ReadoutPulse(500, 400, 0.9, 20e6, 0, Rectangular(), 20)
    p6 = DrivePulse(600, 400, 0.9, 20e6, 0, Gaussian(5), 30)

    ps = PulseSequence(p1, p2, p3, p4, p5, p6)
    n = 70
    for segregated_ps in ps.separate_overlapping_pulses():
        n += 1
        for pulse in segregated_ps:
            pulse.channel = n
    # ps.plot()


def test_pulses_pulse_symbolic_expressions():
    t0 = se_int(0, "t0")
    t = se_int(0, "t")
    p1 = DrivePulse(t0, 400, 0.9, 20e6, 0, Gaussian(5), 10)
    p2 = ReadoutPulse(p1.se_finish + t, 400, 0.9, 20e6, 0, Rectangular(), 30)
    p3 = DrivePulse(p2.se_finish, 400, 0.9, 20e6, 0, Drag(5, 50), 20)
    ps = p1 + p2 + p3

    assert p1.start == 0 and p1.finish == 400
    assert p2.start == 400 and p2.finish == 800
    assert p3.start == 800 and p3.finish == 1200
    assert ps.start == 0 and ps.finish == 1200

    def update(start=0, t_between=0):
        t.value = t_between
        t0.value = start

    update(50, 100)
    assert p1.start == 50 and p1.finish == 450
    assert p2.start == 550 and p2.finish == 950
    assert p3.start == 950 and p3.finish == 1350
    assert ps.start == 50 and ps.finish == 1350


def test_pulses_pulse_pulse_order():
    t0 = se_int(0, "t0")
    t = se_int(0, "t")
    p1 = DrivePulse(t0, 400, 0.9, 20e6, 0, Gaussian(5), 10)
    p2 = ReadoutPulse(p1.se_finish + t, 400, 0.9, 20e6, 0, Rectangular(), 30)
    p3 = DrivePulse(p2.se_finish, 400, 0.9, 20e6, 0, Drag(5, 50), 20)
    ps1 = p1 + p2 + p3
    ps2 = p3 + p1 + p2
    assert ps1 == ps2
    assert hash(ps1) == hash(ps2)


def test_pulses_waveform():
    wf1 = Waveform(np.ones(100))
    wf2 = Waveform(np.zeros(100))
    wf3 = Waveform(np.ones(100))
    assert wf1 != wf2
    assert wf1 == wf3
    np.testing.assert_allclose(wf1.data, wf3.data)
    assert hash(wf1) == hash(str(np.around(np.ones(100), Waveform.DECIMALS) + 0))
    wf1.serial = "Serial works as a tag. The user can set is as desired"
    assert repr(wf1) == wf1.serial


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


def test_pulses_pulseshape_rectangular():
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

    assert pulse.duration == 50
    assert isinstance(pulse.shape, Rectangular)
    assert pulse.shape.name == "Rectangular"
    assert repr(pulse.shape) == "Rectangular()"
    assert isinstance(pulse.shape.envelope_waveform_i(), Waveform)
    assert isinstance(pulse.shape.envelope_waveform_q(), Waveform)
    assert isinstance(pulse.shape.modulated_waveform_i(), Waveform)
    assert isinstance(pulse.shape.modulated_waveform_q(), Waveform)

    sampling_rate = 1
    num_samples = int(pulse.duration / sampling_rate)
    i, q = (
        pulse.amplitude * np.ones(num_samples),
        pulse.amplitude * np.zeros(num_samples),
    )
    global_phase = 2 * np.pi * pulse._if * pulse.start / 1e9  # pulse start, duration and finish are in ns
    mod_i, mod_q = modulate(i, q, num_samples, pulse._if, global_phase + pulse.relative_phase, sampling_rate)

    np.testing.assert_allclose(pulse.shape.envelope_waveform_i(sampling_rate).data, i)
    np.testing.assert_allclose(pulse.shape.envelope_waveform_q(sampling_rate).data, q)
    np.testing.assert_allclose(pulse.shape.modulated_waveform_i(sampling_rate).data, mod_i)
    np.testing.assert_allclose(pulse.shape.modulated_waveform_q(sampling_rate).data, mod_q)

    assert (
        pulse.shape.envelope_waveform_i().serial
        == f"Envelope_Waveform_I(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)})"
    )
    assert (
        pulse.shape.envelope_waveform_q().serial
        == f"Envelope_Waveform_Q(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)})"
    )
    assert (
        pulse.shape.modulated_waveform_i().serial
        == f"Modulated_Waveform_I(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)}, frequency = {format(pulse._if, '_')}, phase = {format(global_phase + pulse.relative_phase, '.6f').rstrip('0').rstrip('.')})"
    )
    assert (
        pulse.shape.modulated_waveform_q().serial
        == f"Modulated_Waveform_Q(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)}, frequency = {format(pulse._if, '_')}, phase = {format(global_phase + pulse.relative_phase, '.6f').rstrip('0').rstrip('.')})"
    )


def test_pulses_pulseshape_gaussian():
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

    assert pulse.duration == 50
    assert isinstance(pulse.shape, Gaussian)
    assert pulse.shape.name == "Gaussian"
    assert pulse.shape.rel_sigma == 5
    assert repr(pulse.shape) == "Gaussian(5)"
    assert isinstance(pulse.shape.envelope_waveform_i(), Waveform)
    assert isinstance(pulse.shape.envelope_waveform_q(), Waveform)
    assert isinstance(pulse.shape.modulated_waveform_i(), Waveform)
    assert isinstance(pulse.shape.modulated_waveform_q(), Waveform)

    sampling_rate = 1
    num_samples = int(pulse.duration / sampling_rate)
    x = np.arange(0, num_samples, 1)
    i = pulse.amplitude * np.exp(
        -(1 / 2) * (((x - (num_samples - 1) / 2) ** 2) / (((num_samples) / pulse.shape.rel_sigma) ** 2))
    )
    q = pulse.amplitude * np.zeros(num_samples)
    global_phase = 2 * np.pi * pulse.frequency * pulse.start / 1e9  # pulse start, duration and finish are in ns
    mod_i, mod_q = modulate(i, q, num_samples, pulse._if, global_phase + pulse.relative_phase, sampling_rate)

    np.testing.assert_allclose(pulse.shape.envelope_waveform_i(sampling_rate).data, i)
    np.testing.assert_allclose(pulse.shape.envelope_waveform_q(sampling_rate).data, q)
    np.testing.assert_allclose(pulse.shape.modulated_waveform_i(sampling_rate).data, mod_i)
    np.testing.assert_allclose(pulse.shape.modulated_waveform_q(sampling_rate).data, mod_q)

    assert (
        pulse.shape.envelope_waveform_i().serial
        == f"Envelope_Waveform_I(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)})"
    )
    assert (
        pulse.shape.envelope_waveform_q().serial
        == f"Envelope_Waveform_Q(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)})"
    )
    assert (
        pulse.shape.modulated_waveform_i().serial
        == f"Modulated_Waveform_I(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)}, frequency = {format(pulse._if, '_')}, phase = {format(global_phase + pulse.relative_phase, '.6f').rstrip('0').rstrip('.')})"
    )
    assert (
        pulse.shape.modulated_waveform_q().serial
        == f"Modulated_Waveform_Q(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)}, frequency = {format(pulse._if, '_')}, phase = {format(global_phase + pulse.relative_phase, '.6f').rstrip('0').rstrip('.')})"
    )


def test_pulses_pulseshape_drag():
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

    assert pulse.duration == 50
    assert isinstance(pulse.shape, Drag)
    assert pulse.shape.name == "Drag"
    assert pulse.shape.rel_sigma == 5
    assert pulse.shape.beta == 0.2
    assert repr(pulse.shape) == "Drag(5, 0.2)"
    assert isinstance(pulse.shape.envelope_waveform_i(), Waveform)
    assert isinstance(pulse.shape.envelope_waveform_q(), Waveform)
    assert isinstance(pulse.shape.modulated_waveform_i(), Waveform)
    assert isinstance(pulse.shape.modulated_waveform_q(), Waveform)

    sampling_rate = 1
    num_samples = int(pulse.duration / 1 * sampling_rate)
    x = np.arange(0, num_samples, 1)
    i = pulse.amplitude * np.exp(
        -(1 / 2) * (((x - (num_samples - 1) / 2) ** 2) / (((num_samples) / pulse.shape.rel_sigma) ** 2))
    )
    q = (
        pulse.shape.beta
        * (-(x - (num_samples - 1) / 2) / ((num_samples / pulse.shape.rel_sigma) ** 2))
        * i
        * sampling_rate
    )
    global_phase = 2 * np.pi * pulse._if * pulse.start / 1e9  # pulse start, duration and finish are in ns
    mod_i, mod_q = modulate(i, q, num_samples, pulse._if, global_phase + pulse.relative_phase, sampling_rate)

    np.testing.assert_allclose(pulse.shape.envelope_waveform_i(sampling_rate).data, i)
    np.testing.assert_allclose(pulse.shape.envelope_waveform_q(sampling_rate).data, q)
    np.testing.assert_allclose(pulse.shape.modulated_waveform_i(sampling_rate).data, mod_i)
    np.testing.assert_allclose(pulse.shape.modulated_waveform_q(sampling_rate).data, mod_q)

    assert (
        pulse.shape.envelope_waveform_i().serial
        == f"Envelope_Waveform_I(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)})"
    )
    assert (
        pulse.shape.envelope_waveform_q().serial
        == f"Envelope_Waveform_Q(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)})"
    )
    assert (
        pulse.shape.modulated_waveform_i().serial
        == f"Modulated_Waveform_I(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)}, frequency = {format(pulse._if, '_')}, phase = {format(global_phase + pulse.relative_phase, '.6f').rstrip('0').rstrip('.')})"
    )
    assert (
        pulse.shape.modulated_waveform_q().serial
        == f"Modulated_Waveform_Q(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)}, frequency = {format(pulse._if, '_')}, phase = {format(global_phase + pulse.relative_phase, '.6f').rstrip('0').rstrip('.')})"
    )


def test_pulses_pulseshape_eq():
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

    target = f"Pulse({pulse.start}, {pulse.duration}, {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, {format(pulse.frequency, '_')}, {format(pulse.relative_phase, '.6f').rstrip('0').rstrip('.')}, {pulse.shape}, {pulse.channel}, {pulse.type}, {pulse.qubit})"
    assert pulse.serial == target
    assert repr(pulse) == target


def test_readout_pulse():
    duration = 2000
    pulse = ReadoutPulse(
        start=0,
        frequency=200_000_000,
        amplitude=1,
        duration=duration,
        relative_phase=0,
        shape=f"Rectangular()",
        channel=11,
    )

    target = f"ReadoutPulse({pulse.start}, {pulse.duration}, {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, {format(pulse.frequency, '_')}, {format(pulse.relative_phase, '.6f').rstrip('0').rstrip('.')}, {pulse.shape}, {pulse.channel}, {pulse.qubit})"
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
            relative_phase=0,
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
            relative_phase=0,
            shape="Gaussian(5)",
            channel=1,
        )
    )
    assert len(sequence.pulses) == 2
    assert len(sequence.qd_pulses) == 2


def test_pulse_sequence__add__():
    sequence = PulseSequence()
    sequence.add(
        Pulse(
            start=0,
            frequency=200_000_000,
            amplitude=0.3,
            duration=60,
            relative_phase=0,
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
            relative_phase=0,
            shape="Gaussian(5)",
            channel=1,
        )
    )
    with pytest.raises(TypeError):
        sequence + 2
    with pytest.raises(TypeError):
        2 + sequence


def test_pulse_sequence__mul__():
    sequence = PulseSequence()
    sequence.add(
        Pulse(
            start=0,
            frequency=200_000_000,
            amplitude=0.3,
            duration=60,
            relative_phase=0,
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
            relative_phase=0,
            shape="Gaussian(5)",
            channel=1,
        )
    )
    with pytest.raises(TypeError):
        sequence * 2.5
    with pytest.raises(TypeError):
        sequence *= 2.5
    with pytest.raises(TypeError):
        sequence *= -1
    with pytest.raises(TypeError):
        sequence * -1
    with pytest.raises(TypeError):
        2.5 * sequence


def test_pulse_sequence_add_readout():
    sequence = PulseSequence()
    sequence.add(
        Pulse(
            start=0,
            frequency=200_000_000,
            amplitude=0.3,
            duration=60,
            relative_phase=0,
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
            relative_phase=0,
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
            relative_phase=0,
            shape="Rectangular()",
            channel=11,
        )
    )
    assert len(sequence.pulses) == 3
    assert len(sequence.ro_pulses) == 1
    assert len(sequence.qd_pulses) == 1
    assert len(sequence.qf_pulses) == 1


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
    assert isinstance(custom_shape_pulse.envelope_waveform_i(), Waveform)
    assert isinstance(custom_shape_pulse.envelope_waveform_q(), Waveform)
    assert isinstance(custom_shape_pulse_old_behaviour.envelope_waveform_q(), Waveform)
    pulse.duration = 2000
    with pytest.raises(ValueError):
        custom_shape_pulse.pulse = pulse
        custom_shape_pulse.envelope_waveform_i()
    with pytest.raises(ValueError):
        custom_shape_pulse.pulse = pulse
        custom_shape_pulse.envelope_waveform_q()
