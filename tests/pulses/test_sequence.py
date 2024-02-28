from qibolab.pulses import Drag, Gaussian, Pulse, PulseSequence, PulseType, Rectangular


def test_add_readout():
    sequence = PulseSequence()
    sequence.append(
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

    sequence.append(
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

    sequence.append(
        Pulse(
            start=128,
            frequency=20_000_000,
            amplitude=0.9,
            duration=2000,
            relative_phase=0,
            shape="Rectangular()",
            channel=11,
            type=PulseType.READOUT,
        )
    )
    assert len(sequence) == 3
    assert len(sequence.ro_pulses) == 1
    assert len(sequence.qd_pulses) == 1
    assert len(sequence.qf_pulses) == 1


def test_separate_overlapping_pulses():
    p1 = Pulse(0, 400, 0.9, 20e6, 0, Gaussian(5), 10)
    p2 = Pulse(100, 400, 0.9, 20e6, 0, Rectangular(), qubit=30, type=PulseType.READOUT)
    p3 = Pulse(300, 400, 0.9, 20e6, 0, Drag(5, 50), 20)
    p4 = Pulse(400, 400, 0.9, 20e6, 0, Drag(5, 50), 30)
    p5 = Pulse(500, 400, 0.9, 20e6, 0, Rectangular(), qubit=20, type=PulseType.READOUT)
    p6 = Pulse(600, 400, 0.9, 20e6, 0, Gaussian(5), 30)

    ps = PulseSequence([p1, p2, p3, p4, p5, p6])
    n = 70
    for segregated_ps in ps.separate_overlapping_pulses():
        n += 1
        for pulse in segregated_ps:
            pulse.channel = n


def test_get_qubit_pulses():
    p1 = Pulse(0, 400, 0.9, 20e6, 0, Gaussian(5), 10, qubit=0)
    p2 = Pulse(
        100,
        400,
        0.9,
        20e6,
        0,
        Rectangular(),
        channel=30,
        qubit=0,
        type=PulseType.READOUT,
    )
    p3 = Pulse(300, 400, 0.9, 20e6, 0, Drag(5, 50), 20, qubit=1)
    p4 = Pulse(400, 400, 0.9, 20e6, 0, Drag(5, 50), 30, qubit=1)
    p5 = Pulse(
        500,
        400,
        0.9,
        20e6,
        0,
        Rectangular(),
        channel=30,
        qubit=1,
        type=PulseType.READOUT,
    )
    p6 = Pulse.flux(600, 400, 0.9, Rectangular(), channel=40, qubit=1)
    p7 = Pulse.flux(900, 400, 0.9, Rectangular(), channel=40, qubit=2)

    ps = PulseSequence([p1, p2, p3, p4, p5, p6, p7])
    assert ps.qubits == [0, 1, 2]
    assert len(ps.get_qubit_pulses(0)) == 2
    assert len(ps.get_qubit_pulses(1)) == 4
    assert len(ps.get_qubit_pulses(2)) == 1
    assert len(ps.get_qubit_pulses(0, 1)) == 6


def test_pulses_overlap():
    p1 = Pulse(0, 400, 0.9, 20e6, 0, Gaussian(5), 10)
    p2 = Pulse(100, 400, 0.9, 20e6, 0, Rectangular(), 30, type=PulseType.READOUT)
    p3 = Pulse(300, 400, 0.9, 20e6, 0, Drag(5, 50), 20)
    p4 = Pulse(400, 400, 0.9, 20e6, 0, Drag(5, 50), 30)
    p5 = Pulse(500, 400, 0.9, 20e6, 0, Rectangular(), 20, type=PulseType.READOUT)
    p6 = Pulse(600, 400, 0.9, 20e6, 0, Gaussian(5), 30)

    ps = PulseSequence([p1, p2, p3, p4, p5, p6])
    assert ps.pulses_overlap
    assert not ps.get_channel_pulses(10).pulses_overlap
    assert ps.get_channel_pulses(20).pulses_overlap
    assert ps.get_channel_pulses(30).pulses_overlap


def test_get_channel_pulses():
    p1 = Pulse(0, 400, 0.9, 20e6, 0, Gaussian(5), 10)
    p2 = Pulse(100, 400, 0.9, 20e6, 0, Rectangular(), 30, type=PulseType.READOUT)
    p3 = Pulse(300, 400, 0.9, 20e6, 0, Drag(5, 50), 20)
    p4 = Pulse(400, 400, 0.9, 20e6, 0, Drag(5, 50), 30)
    p5 = Pulse(500, 400, 0.9, 20e6, 0, Rectangular(), 20, type=PulseType.READOUT)
    p6 = Pulse(600, 400, 0.9, 20e6, 0, Gaussian(5), 30)

    ps = PulseSequence([p1, p2, p3, p4, p5, p6])
    assert ps.channels == [10, 20, 30]
    assert len(ps.get_channel_pulses(10)) == 1
    assert len(ps.get_channel_pulses(20)) == 2
    assert len(ps.get_channel_pulses(30)) == 3
    assert len(ps.get_channel_pulses(20, 30)) == 5


def test_start_finish():
    p1 = Pulse(20, 40, 0.9, 200e6, 0, Drag(5, 1), 1, PulseType.DRIVE)
    p2 = Pulse(60, 1000, 0.9, 20e6, 0, Rectangular(), 2, PulseType.READOUT)
    ps = PulseSequence([p1]) + [p2]
    assert ps.start == p1.start
    assert ps.finish == p2.finish

    p1.start = None
    assert p1.finish is None
    p2.duration = None
    assert p2.finish is None


def test_init():
    p1 = Pulse(400, 40, 0.9, 100e6, 0, Drag(5, 1), 3, PulseType.DRIVE)
    p2 = Pulse(500, 40, 0.9, 100e6, 0, Drag(5, 1), 2, PulseType.DRIVE)
    p3 = Pulse(600, 40, 0.9, 100e6, 0, Drag(5, 1), 1, PulseType.DRIVE)

    ps = PulseSequence()
    assert type(ps) == PulseSequence

    ps = PulseSequence([p1, p2, p3])
    assert len(ps) == 3
    assert ps[0] == p1
    assert ps[1] == p2
    assert ps[2] == p3

    other_ps = PulseSequence([p1, p2, p3])
    assert len(other_ps) == 3
    assert other_ps[0] == p1
    assert other_ps[1] == p2
    assert other_ps[2] == p3

    plist = [p1, p2, p3]
    n = 0
    for pulse in ps:
        assert plist[n] == pulse
        n += 1


def test_operators():
    ps = PulseSequence()
    ps += [Pulse(800, 200, 0.9, 20e6, 0, Rectangular(), 1, type=PulseType.READOUT)]
    ps = ps + [Pulse(800, 200, 0.9, 20e6, 0, Rectangular(), 2, type=PulseType.READOUT)]
    ps = [Pulse(800, 200, 0.9, 20e6, 0, Rectangular(), 3, type=PulseType.READOUT)] + ps

    p4 = Pulse(100, 40, 0.9, 50e6, 0, Gaussian(5), 3, PulseType.DRIVE)
    p5 = Pulse(200, 40, 0.9, 50e6, 0, Gaussian(5), 2, PulseType.DRIVE)
    p6 = Pulse(300, 40, 0.9, 50e6, 0, Gaussian(5), 1, PulseType.DRIVE)

    another_ps = PulseSequence()
    another_ps.append(p4)
    another_ps.extend([p5, p6])

    assert another_ps[0] == p4
    assert another_ps[1] == p5
    assert another_ps[2] == p6

    ps += another_ps

    assert len(ps) == 6
    assert p5 in ps

    # ps.plot()

    p7 = Pulse(600, 40, 0.9, 100e6, 0, Drag(5, 1), 1, PulseType.DRIVE)
    yet_another_ps = PulseSequence([p7])
    assert len(yet_another_ps) == 1
    yet_another_ps *= 3
    assert len(yet_another_ps) == 3
    yet_another_ps *= 3
    assert len(yet_another_ps) == 9

    p8 = Pulse(600, 40, 0.9, 100e6, 0, Drag(5, 1), 1, PulseType.DRIVE)
    p9 = Pulse(600, 40, 0.9, 100e6, 0, Drag(5, 1), 2, PulseType.DRIVE)
    and_yet_another_ps = 2 * PulseSequence([p9]) + [p8] * 3
    assert len(and_yet_another_ps) == 5
