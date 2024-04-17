from qibolab.pulses import (
    Delay,
    Drag,
    Gaussian,
    Pulse,
    PulseSequence,
    PulseType,
    Rectangular,
)


def test_add_readout():
    sequence = PulseSequence()
    sequence.append(
        Pulse(
            frequency=200_000_000,
            amplitude=0.3,
            duration=60,
            relative_phase=0,
            envelope=Gaussian(rel_sigma=0.2),
            channel="1",
        )
    )
    sequence.append(Delay(duration=4, channel="1"))
    sequence.append(
        Pulse(
            frequency=200_000_000,
            amplitude=0.3,
            duration=60,
            relative_phase=0,
            envelope=Drag(rel_sigma=0.2, beta=2),
            channel="1",
            type=PulseType.FLUX,
        )
    )
    sequence.append(Delay(duration=4, channel="1"))
    sequence.append(
        Pulse(
            frequency=20_000_000,
            amplitude=0.9,
            duration=2000,
            relative_phase=0,
            envelope=Rectangular(),
            channel="11",
            type=PulseType.READOUT,
        )
    )
    assert len(sequence) == 5
    assert len(sequence.ro_pulses) == 1
    assert len(sequence.qd_pulses) == 1
    assert len(sequence.qf_pulses) == 1


def test_get_qubit_pulses():
    p1 = Pulse(
        duration=400,
        amplitude=0.9,
        frequency=20e6,
        envelope=Gaussian(rel_sigma=0.2),
        relative_phase=10,
        qubit=0,
    )
    p2 = Pulse(
        duration=400,
        amplitude=0.9,
        frequency=20e6,
        envelope=Rectangular(),
        channel="30",
        qubit=0,
        type=PulseType.READOUT,
    )
    p3 = Pulse(
        duration=400,
        amplitude=0.9,
        frequency=20e6,
        envelope=Drag(rel_sigma=0.2, beta=50),
        relative_phase=20,
        qubit=1,
    )
    p4 = Pulse(
        duration=400,
        amplitude=0.9,
        frequency=20e6,
        envelope=Drag(rel_sigma=0.2, beta=50),
        relative_phase=30,
        qubit=1,
    )
    p5 = Pulse(
        duration=400,
        amplitude=0.9,
        frequency=20e6,
        envelope=Rectangular(),
        channel="30",
        qubit=1,
        type=PulseType.READOUT,
    )
    p6 = Pulse.flux(
        duration=400, amplitude=0.9, envelope=Rectangular(), channel="40", qubit=1
    )
    p7 = Pulse.flux(
        duration=400, amplitude=0.9, envelope=Rectangular(), channel="40", qubit=2
    )

    ps = PulseSequence([p1, p2, p3, p4, p5, p6, p7])
    assert ps.qubits == [0, 1, 2]
    assert len(ps.get_qubit_pulses(0)) == 2
    assert len(ps.get_qubit_pulses(1)) == 4
    assert len(ps.get_qubit_pulses(2)) == 1
    assert len(ps.get_qubit_pulses(0, 1)) == 6


def test_get_channel_pulses():
    p1 = Pulse(
        duration=400,
        frequency=0.9,
        amplitude=20e6,
        envelope=Gaussian(rel_sigma=0.2),
        channel="10",
    )
    p2 = Pulse(
        duration=400,
        frequency=0.9,
        amplitude=20e6,
        envelope=Rectangular(),
        channel="30",
        type=PulseType.READOUT,
    )
    p3 = Pulse(
        duration=400,
        frequency=0.9,
        amplitude=20e6,
        envelope=Drag(rel_sigma=0.2, beta=5),
        channel="20",
    )
    p4 = Pulse(
        duration=400,
        frequency=0.9,
        amplitude=20e6,
        envelope=Drag(rel_sigma=0.2, beta=5),
        channel="30",
    )
    p5 = Pulse(
        duration=400,
        frequency=0.9,
        amplitude=20e6,
        envelope=Rectangular(),
        channel="20",
        type=PulseType.READOUT,
    )
    p6 = Pulse(
        duration=400,
        frequency=0.9,
        amplitude=20e6,
        envelope=Gaussian(rel_sigma=0.2),
        channel="30",
    )

    ps = PulseSequence([p1, p2, p3, p4, p5, p6])
    assert sorted(ps.channels) == ["10", "20", "30"]
    assert len(ps.get_channel_pulses("10")) == 1
    assert len(ps.get_channel_pulses("20")) == 2
    assert len(ps.get_channel_pulses("30")) == 3
    assert len(ps.get_channel_pulses("20", "30")) == 5


def test_sequence_duration():
    p0 = Delay(duration=20, channel="1")
    p1 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=200e6,
        envelope=Drag(rel_sigma=0.2, beta=1),
        channel="1",
        type=PulseType.DRIVE,
    )
    p2 = Pulse(
        duration=1000,
        amplitude=0.9,
        frequency=20e6,
        envelope=Rectangular(),
        channel="1",
        type=PulseType.READOUT,
    )
    ps = PulseSequence([p0, p1]) + [p2]
    assert ps.duration == 20 + 40 + 1000
    ps[-1] = p2.model_copy(update={"channel": "2"})
    assert ps.duration == 1000


def test_init():
    p1 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=100e6,
        envelope=Drag(rel_sigma=0.2, beta=1),
        channel="3",
        type=PulseType.DRIVE,
    )
    p2 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=100e6,
        envelope=Drag(rel_sigma=0.2, beta=1),
        channel="2",
        type=PulseType.DRIVE,
    )
    p3 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=100e6,
        envelope=Drag(rel_sigma=0.2, beta=1),
        channel="1",
        type=PulseType.DRIVE,
    )

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
    ps += [
        Pulse(
            duration=200,
            amplitude=0.9,
            frequency=20e6,
            envelope=Rectangular(),
            channel="3",
            type=PulseType.DRIVE,
        )
    ]
    ps = ps + [
        Pulse(
            duration=200,
            amplitude=0.9,
            frequency=20e6,
            envelope=Rectangular(),
            channel="2",
            type=PulseType.DRIVE,
        )
    ]
    ps = [
        Pulse(
            duration=200,
            amplitude=0.9,
            frequency=20e6,
            envelope=Rectangular(),
            channel="3",
            type=PulseType.DRIVE,
        )
    ] + ps

    p4 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=50e6,
        envelope=Gaussian(rel_sigma=0.2),
        channel="3",
        type=PulseType.DRIVE,
    )
    p5 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=50e6,
        envelope=Gaussian(rel_sigma=0.2),
        channel="2",
        type=PulseType.DRIVE,
    )
    p6 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=50e6,
        envelope=Gaussian(rel_sigma=0.2),
        channel="1",
        type=PulseType.DRIVE,
    )

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

    p7 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=100e6,
        envelope=Drag(rel_sigma=0.2, beta=1),
        channel="1",
        type=PulseType.DRIVE,
    )
    yet_another_ps = PulseSequence([p7])
    assert len(yet_another_ps) == 1
    yet_another_ps *= 3
    assert len(yet_another_ps) == 3
    yet_another_ps *= 3
    assert len(yet_another_ps) == 9

    p8 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=100e6,
        envelope=Drag(rel_sigma=0.2, beta=1),
        channel="1",
        type=PulseType.DRIVE,
    )
    p9 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=100e6,
        envelope=Drag(rel_sigma=0.2, beta=1),
        channel="2",
        type=PulseType.DRIVE,
    )
    and_yet_another_ps = 2 * PulseSequence([p9]) + [p8] * 3
    assert len(and_yet_another_ps) == 5
