from collections import defaultdict

from qibolab.pulses import (
    Delay,
    Drag,
    Gaussian,
    Pulse,
    PulseSequence,
    PulseType,
    Rectangular,
)


def test_init():
    sequence = PulseSequence()
    assert isinstance(sequence, defaultdict)
    assert len(sequence) == 0


def test_default_factory():
    sequence = PulseSequence()
    some = sequence["some channel"]
    assert isinstance(some, list)
    assert len(some) == 0


def test_ro_pulses():
    Pulse(
        amplitude=0.3,
        duration=60,
        relative_phase=0,
        envelope=Gaussian(rel_sigma=0.2),
    )
    sequence = PulseSequence()
    sequence["ch1"].append(
        Pulse(
            amplitude=0.3,
            duration=60,
            relative_phase=0,
            envelope=Gaussian(rel_sigma=0.2),
        )
    )
    sequence["ch2"].append(Delay(duration=4))
    sequence["ch2"].append(
        Pulse(
            amplitude=0.3,
            duration=60,
            relative_phase=0,
            envelope=Drag(rel_sigma=0.2, beta=2),
            type=PulseType.FLUX,
        )
    )
    sequence["ch3"].append(Delay(duration=4))
    ro_pulse = Pulse(
        amplitude=0.9,
        duration=2000,
        relative_phase=0,
        envelope=Rectangular(),
        type=PulseType.READOUT,
    )
    sequence["ch3"].append(ro_pulse)
    assert set(sequence.keys()) == {"ch1", "ch2", "ch3"}
    assert sum(len(pulses) for pulses in sequence.values()) == 5
    assert len(sequence.ro_pulses) == 1
    assert sequence.ro_pulses[0] == ro_pulse


def test_durations():
    sequence = PulseSequence()
    sequence["ch1"].append(Delay(duration=20))
    sequence["ch1"].append(
        Pulse(
            duration=1000,
            amplitude=0.9,
            envelope=Rectangular(),
            type=PulseType.READOUT,
        )
    )
    sequence["ch2"].append(
        Pulse(
            duration=40,
            amplitude=0.9,
            envelope=Drag(rel_sigma=0.2, beta=1),
            type=PulseType.DRIVE,
        )
    )
    assert sequence.channel_duration("ch1") == 20 + 1000
    assert sequence.channel_duration("ch2") == 40
    assert sequence.duration == 20 + 1000

    sequence["ch2"].append(
        Pulse(
            duration=1200,
            amplitude=0.9,
            envelope=Rectangular(),
            type=PulseType.READOUT,
        )
    )
    assert sequence.duration == 40 + 1200


def test_extend():
    sequence1 = PulseSequence()
    sequence1["ch1"].append(
        Pulse(
            duration=40,
            amplitude=0.9,
            envelope=Drag(rel_sigma=0.2, beta=1),
        )
    )

    sequence2 = PulseSequence()
    sequence2["ch2"].append(
        Pulse(
            duration=60,
            amplitude=0.9,
            envelope=Drag(rel_sigma=0.2, beta=1),
        )
    )

    sequence1.extend(sequence2)
    assert set(sequence1.keys()) == {"ch1", "ch2"}
    assert len(sequence1["ch1"]) == 1
    assert len(sequence1["ch2"]) == 1
    assert sequence1.duration == 60

    sequence3 = PulseSequence()
    sequence3["ch2"].append(
        Pulse(
            duration=80,
            amplitude=0.9,
            envelope=Drag(rel_sigma=0.2, beta=1),
        )
    )
    sequence3["ch3"].append(
        Pulse(
            duration=100,
            amplitude=0.9,
            envelope=Drag(rel_sigma=0.2, beta=1),
        )
    )

    sequence1.extend(sequence3)
    assert set(sequence1.keys()) == {"ch1", "ch2", "ch3"}
    assert len(sequence1["ch1"]) == 1
    assert len(sequence1["ch2"]) == 2
    assert len(sequence1["ch3"]) == 2
    assert isinstance(sequence1["ch3"][0], Delay)
    assert sequence1.duration == 60 + 100
    assert sequence1.channel_duration("ch1") == 40
    assert sequence1.channel_duration("ch2") == 60 + 80
    assert sequence1.channel_duration("ch3") == 60 + 100
