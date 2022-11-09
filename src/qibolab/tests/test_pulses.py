"""Tests ``pulses.py``."""
import numpy as np
import pytest

from qibolab.pulses import (
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
    SplitPulse,
    Waveform,
)
from qibolab.symbolic import intSymbolicExpression as se_int


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
            duration=0,  # duration should be > 0
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


def test_pulses_pulse_serial():
    p11 = Pulse(0, 40, 0.9, 50_000_000, 0, Gaussian(5), 0, PulseType.DRIVE)
    assert p11.serial == "Pulse(0, 40, 0.9, 50_000_000, 0, Gaussian(5), 0, PulseType.DRIVE, 0)"
    assert repr(p11) == p11.serial


def test_pulses_pulseshape_sampling_rate():
    p12 = Pulse(0, 40, 0.9, 100e6, 0, Rectangular(), 0, PulseType.DRIVE)
    p13 = Pulse(0, 40, 0.9, 100e6, 0, Gaussian(5), 0, PulseType.DRIVE)
    p14 = Pulse(0, 40, 0.9, 100e6, 0, Drag(5, 1), 0, PulseType.DRIVE)

    tmp = PulseShape.SAMPLING_RATE
    PulseShape.SAMPLING_RATE = 1e9
    # p12.plot()
    # p13.plot()
    # p14.plot()
    PulseShape.SAMPLING_RATE = tmp

    tmp = PulseShape.SAMPLING_RATE
    PulseShape.SAMPLING_RATE = 100e9
    # p12.plot()
    # p13.plot()
    # p14.plot()
    PulseShape.SAMPLING_RATE = tmp


def test_pulses_pulseshape_drag_shape():
    tmp = PulseShape.SAMPLING_RATE
    dp = Pulse(0, 2, 1, 4e9, 0, Drag(2, 1), 0, PulseType.DRIVE)
    PulseShape.SAMPLING_RATE = 100e9
    # dp.plot()
    PulseShape.SAMPLING_RATE = tmp
    # envelope i & envelope q should cross nearly at 0 and at 2


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

    fp = FluxPulse(start=0, duration=300, amplitude=0.9, relative_phase=0.0, shape=Rectangular(), channel=0, qubit=0)
    assert repr(fp) == "FluxPulse(0, 300, 0.9, 0, Rectangular(), 0, 0)"


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
    assert len(sp.envelope_waveform_i) == b - a
    assert len(sp.envelope_waveform_q) == b - a
    assert len(sp.modulated_waveform_i) == b - a
    assert len(sp.modulated_waveform_q) == b - a


def test_pulses_pulsesequence_init():
    p1 = Pulse(600, 40, 0.9, 100e6, 0, Drag(5, 1), 1, PulseType.DRIVE)
    p2 = Pulse(500, 40, 0.9, 100e6, 0, Drag(5, 1), 2, PulseType.DRIVE)
    p3 = Pulse(400, 40, 0.9, 100e6, 0, Drag(5, 1), 3, PulseType.DRIVE)

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

    p4 = Pulse(300, 40, 0.9, 50e6, 0, Gaussian(5), 1, PulseType.DRIVE)
    p5 = Pulse(200, 40, 0.9, 50e6, 0, Gaussian(5), 2, PulseType.DRIVE)
    p6 = Pulse(100, 40, 0.9, 50e6, 0, Gaussian(5), 3, PulseType.DRIVE)

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
    i: np.ndarray, q: np.ndarray, num_samples: int, frequency: int, phase: float
):  #  -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(num_samples) / PulseShape.SAMPLING_RATE
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
    assert isinstance(pulse.shape.envelope_waveform_i, Waveform)
    assert isinstance(pulse.shape.envelope_waveform_q, Waveform)
    assert isinstance(pulse.shape.modulated_waveform_i, Waveform)
    assert isinstance(pulse.shape.modulated_waveform_q, Waveform)

    num_samples = int(pulse.duration / 1e9 * PulseShape.SAMPLING_RATE)
    i, q = pulse.amplitude * np.ones(num_samples), pulse.amplitude * np.zeros(num_samples)
    global_phase = 2 * np.pi * pulse.frequency * pulse.start / 1e9  # pulse start, duration and finish are in ns
    mod_i, mod_q = modulate(i, q, num_samples, pulse.frequency, global_phase + pulse.relative_phase)

    np.testing.assert_allclose(pulse.shape.envelope_waveform_i.data, i)
    np.testing.assert_allclose(pulse.shape.envelope_waveform_q.data, q)
    np.testing.assert_allclose(pulse.shape.modulated_waveform_i.data, mod_i)
    np.testing.assert_allclose(pulse.shape.modulated_waveform_q.data, mod_q)

    assert (
        pulse.shape.envelope_waveform_i.serial
        == f"Envelope_Waveform_I(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)})"
    )
    assert (
        pulse.shape.envelope_waveform_q.serial
        == f"Envelope_Waveform_Q(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)})"
    )
    assert (
        pulse.shape.modulated_waveform_i.serial
        == f"Modulated_Waveform_I(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)}, frequency = {format(pulse.frequency, '_')}, phase = {format(global_phase + pulse.relative_phase, '.6f').rstrip('0').rstrip('.')})"
    )
    assert (
        pulse.shape.modulated_waveform_q.serial
        == f"Modulated_Waveform_Q(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)}, frequency = {format(pulse.frequency, '_')}, phase = {format(global_phase + pulse.relative_phase, '.6f').rstrip('0').rstrip('.')})"
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
    assert isinstance(pulse.shape.envelope_waveform_i, Waveform)
    assert isinstance(pulse.shape.envelope_waveform_q, Waveform)
    assert isinstance(pulse.shape.modulated_waveform_i, Waveform)
    assert isinstance(pulse.shape.modulated_waveform_q, Waveform)

    num_samples = int(pulse.duration / 1e9 * PulseShape.SAMPLING_RATE)
    x = np.arange(0, num_samples, 1)
    i = pulse.amplitude * np.exp(
        -(1 / 2) * (((x - (num_samples - 1) / 2) ** 2) / (((num_samples) / pulse.shape.rel_sigma) ** 2))
    )
    q = pulse.amplitude * np.zeros(num_samples)
    global_phase = 2 * np.pi * pulse.frequency * pulse.start / 1e9  # pulse start, duration and finish are in ns
    mod_i, mod_q = modulate(i, q, num_samples, pulse.frequency, global_phase + pulse.relative_phase)

    np.testing.assert_allclose(pulse.shape.envelope_waveform_i.data, i)
    np.testing.assert_allclose(pulse.shape.envelope_waveform_q.data, q)
    np.testing.assert_allclose(pulse.shape.modulated_waveform_i.data, mod_i)
    np.testing.assert_allclose(pulse.shape.modulated_waveform_q.data, mod_q)

    assert (
        pulse.shape.envelope_waveform_i.serial
        == f"Envelope_Waveform_I(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)})"
    )
    assert (
        pulse.shape.envelope_waveform_q.serial
        == f"Envelope_Waveform_Q(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)})"
    )
    assert (
        pulse.shape.modulated_waveform_i.serial
        == f"Modulated_Waveform_I(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)}, frequency = {format(pulse.frequency, '_')}, phase = {format(global_phase + pulse.relative_phase, '.6f').rstrip('0').rstrip('.')})"
    )
    assert (
        pulse.shape.modulated_waveform_q.serial
        == f"Modulated_Waveform_Q(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)}, frequency = {format(pulse.frequency, '_')}, phase = {format(global_phase + pulse.relative_phase, '.6f').rstrip('0').rstrip('.')})"
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
    assert isinstance(pulse.shape.envelope_waveform_i, Waveform)
    assert isinstance(pulse.shape.envelope_waveform_q, Waveform)
    assert isinstance(pulse.shape.modulated_waveform_i, Waveform)
    assert isinstance(pulse.shape.modulated_waveform_q, Waveform)

    num_samples = int(pulse.duration / 1e9 * PulseShape.SAMPLING_RATE)
    x = np.arange(0, num_samples, 1)
    i = pulse.amplitude * np.exp(
        -(1 / 2) * (((x - (num_samples - 1) / 2) ** 2) / (((num_samples) / pulse.shape.rel_sigma) ** 2))
    )
    q = (
        pulse.shape.beta
        * (-(x - (num_samples - 1) / 2) / ((num_samples / pulse.shape.rel_sigma) ** 2))
        * i
        * PulseShape.SAMPLING_RATE
        / 1e9
    )
    global_phase = 2 * np.pi * pulse.frequency * pulse.start / 1e9  # pulse start, duration and finish are in ns
    mod_i, mod_q = modulate(i, q, num_samples, pulse.frequency, global_phase + pulse.relative_phase)

    np.testing.assert_allclose(pulse.shape.envelope_waveform_i.data, i)
    np.testing.assert_allclose(pulse.shape.envelope_waveform_q.data, q)
    np.testing.assert_allclose(pulse.shape.modulated_waveform_i.data, mod_i)
    np.testing.assert_allclose(pulse.shape.modulated_waveform_q.data, mod_q)

    assert (
        pulse.shape.envelope_waveform_i.serial
        == f"Envelope_Waveform_I(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)})"
    )
    assert (
        pulse.shape.envelope_waveform_q.serial
        == f"Envelope_Waveform_Q(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)})"
    )
    assert (
        pulse.shape.modulated_waveform_i.serial
        == f"Modulated_Waveform_I(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)}, frequency = {format(pulse.frequency, '_')}, phase = {format(global_phase + pulse.relative_phase, '.6f').rstrip('0').rstrip('.')})"
    )
    assert (
        pulse.shape.modulated_waveform_q.serial
        == f"Modulated_Waveform_Q(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)}, frequency = {format(pulse.frequency, '_')}, phase = {format(global_phase + pulse.relative_phase, '.6f').rstrip('0').rstrip('.')})"
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
