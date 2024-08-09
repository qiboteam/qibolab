from qibolab.pulses import Delay, Drag, Gaussian, Pulse, PulseSequence, Rectangular


def test_init():
    sequence = PulseSequence()
    assert len(sequence) == 0


def test_init_with_iterable():
    seq = PulseSequence(
        [
            ("some channel", p)
            for p in [
                Pulse(duration=20, amplitude=0.1, envelope=Gaussian(rel_sigma=3)),
                Pulse(duration=30, amplitude=0.5, envelope=Gaussian(rel_sigma=3)),
            ]
        ]
        + [
            (
                "other channel",
                Pulse(duration=40, amplitude=0.2, envelope=Gaussian(rel_sigma=3)),
            )
        ]
        + [
            ("chanel #5", p)
            for p in [
                Pulse(duration=45, amplitude=1.0, envelope=Gaussian(rel_sigma=3)),
                Pulse(duration=50, amplitude=0.7, envelope=Gaussian(rel_sigma=3)),
                Pulse(duration=60, amplitude=-0.65, envelope=Gaussian(rel_sigma=3)),
            ]
        ]
    )

    assert len(seq) == 6
    assert set(seq.channels) == {"some channel", "other channel", "chanel #5"}
    assert len(list(seq.channel("some channel"))) == 2
    assert len(list(seq.channel("other channel"))) == 1
    assert len(list(seq.channel("chanel #5"))) == 3


def test_durations():
    sequence = PulseSequence()
    sequence.append(("ch1/probe", Delay(duration=20)))
    sequence.append(
        ("ch1/probe", Pulse(duration=1000, amplitude=0.9, envelope=Rectangular()))
    )
    sequence.append(
        (
            "ch2/drive",
            Pulse(duration=40, amplitude=0.9, envelope=Drag(rel_sigma=0.2, beta=1)),
        )
    )
    assert sequence.channel_duration("ch1/probe") == 20 + 1000
    assert sequence.channel_duration("ch2/drive") == 40
    assert sequence.duration == 20 + 1000

    sequence.append(
        ("ch2/drive", Pulse(duration=1200, amplitude=0.9, envelope=Rectangular()))
    )
    assert sequence.duration == 40 + 1200


def test_concatenate():
    sequence1 = PulseSequence(
        [
            (
                "ch1",
                Pulse(duration=40, amplitude=0.9, envelope=Drag(rel_sigma=0.2, beta=1)),
            )
        ]
    )
    sequence2 = PulseSequence(
        [
            (
                "ch2",
                Pulse(duration=60, amplitude=0.9, envelope=Drag(rel_sigma=0.2, beta=1)),
            )
        ]
    )

    sequence1.concatenate(sequence2)
    assert set(sequence1.channels) == {"ch1", "ch2"}
    assert len(list(sequence1.channel("ch1"))) == 1
    assert len(list(sequence1.channel("ch2"))) == 1
    assert sequence1.duration == 60

    sequence3 = PulseSequence(
        [
            (
                "ch2",
                Pulse(duration=80, amplitude=0.9, envelope=Drag(rel_sigma=0.2, beta=1)),
            ),
            (
                "ch3",
                Pulse(
                    duration=100, amplitude=0.9, envelope=Drag(rel_sigma=0.2, beta=1)
                ),
            ),
        ]
    )

    sequence1.concatenate(sequence3)
    assert sequence1.channels == {"ch1", "ch2", "ch3"}
    assert len(list(sequence1.channel("ch1"))) == 1
    assert len(list(sequence1.channel("ch2"))) == 2
    assert len(list(sequence1.channel("ch3"))) == 2
    assert isinstance(next(iter(sequence1.channel("ch3"))), Delay)
    assert sequence1.duration == 60 + 100
    assert sequence1.channel_duration("ch1") == 40
    assert sequence1.channel_duration("ch2") == 60 + 80
    assert sequence1.channel_duration("ch3") == 60 + 100


def test_copy():
    sequence = PulseSequence(
        [
            (
                "ch1",
                Pulse(duration=40, amplitude=0.9, envelope=Drag(rel_sigma=0.2, beta=1)),
            ),
            (
                "ch2",
                Pulse(duration=60, amplitude=0.9, envelope=Drag(rel_sigma=0.2, beta=1)),
            ),
            (
                "ch2",
                Pulse(duration=80, amplitude=0.9, envelope=Drag(rel_sigma=0.2, beta=1)),
            ),
        ]
    )

    sequence_copy = sequence.copy()
    assert isinstance(sequence_copy, PulseSequence)
    assert sequence_copy == sequence

    sequence_copy.append(
        (
            "ch3",
            Pulse(duration=100, amplitude=0.9, envelope=Drag(rel_sigma=0.2, beta=1)),
        )
    )
    assert "ch3" not in sequence
