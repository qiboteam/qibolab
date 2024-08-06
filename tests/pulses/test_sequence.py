from collections import defaultdict

from qibolab.pulses import Delay, Drag, Gaussian, Pulse, PulseSequence, Rectangular


def test_init():
    sequence = PulseSequence()
    assert isinstance(sequence, defaultdict)
    assert len(sequence) == 0


def test_init_with_dict():
    seq_dict = {
        "some channel": [
            Pulse(duration=20, amplitude=0.1, envelope=Gaussian(rel_sigma=3)),
            Pulse(duration=30, amplitude=0.5, envelope=Gaussian(rel_sigma=3)),
        ],
        "other channel": [
            Pulse(duration=40, amplitude=0.2, envelope=Gaussian(rel_sigma=3))
        ],
        "chanel #5": [
            Pulse(duration=45, amplitude=1.0, envelope=Gaussian(rel_sigma=3)),
            Pulse(duration=50, amplitude=0.7, envelope=Gaussian(rel_sigma=3)),
            Pulse(duration=60, amplitude=-0.65, envelope=Gaussian(rel_sigma=3)),
        ],
    }
    seq = PulseSequence(seq_dict)

    assert len(seq) == 3
    assert set(seq.keys()) == set(seq_dict.keys())
    assert len(seq["some channel"]) == 2
    assert len(seq["other channel"]) == 1
    assert len(seq["chanel #5"]) == 3


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
    sequence["ch2/flux"].append(Delay(duration=4))
    sequence["ch2/flux"].append(
        Pulse(
            amplitude=0.3,
            duration=60,
            relative_phase=0,
            envelope=Drag(rel_sigma=0.2, beta=2),
        )
    )
    sequence["ch3/readout"].append(Delay(duration=4))
    ro_pulse = Pulse(
        amplitude=0.9,
        duration=2000,
        relative_phase=0,
        envelope=Rectangular(),
    )
    sequence["ch3/readout"].append(ro_pulse)
    assert set(sequence.keys()) == {"ch1", "ch2/flux", "ch3/readout"}
    assert sum(len(pulses) for pulses in sequence.values()) == 5
    assert len(sequence.probe_pulses) == 1
    assert sequence.probe_pulses[0] == ro_pulse


def test_durations():
    sequence = PulseSequence()
    sequence["ch1/readout"].append(Delay(duration=20))
    sequence["ch1/readout"].append(
        Pulse(
            duration=1000,
            amplitude=0.9,
            envelope=Rectangular(),
        )
    )
    sequence["ch2/drive"].append(
        Pulse(
            duration=40,
            amplitude=0.9,
            envelope=Drag(rel_sigma=0.2, beta=1),
        )
    )
    assert sequence.channel_duration("ch1/drive") == 20 + 1000
    assert sequence.channel_duration("ch2/readout") == 40
    assert sequence.duration == 20 + 1000

    sequence["ch2/readout"].append(
        Pulse(
            duration=1200,
            amplitude=0.9,
            envelope=Rectangular(),
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


def test_copy():
    sequence = PulseSequence()
    sequence["ch1"].append(
        Pulse(
            duration=40,
            amplitude=0.9,
            envelope=Drag(rel_sigma=0.2, beta=1),
        )
    )

    sequence["ch2"].append(
        Pulse(
            duration=60,
            amplitude=0.9,
            envelope=Drag(rel_sigma=0.2, beta=1),
        )
    )
    sequence["ch2"].append(
        Pulse(
            duration=80,
            amplitude=0.9,
            envelope=Drag(rel_sigma=0.2, beta=1),
        )
    )

    sequence_copy = sequence.copy()
    assert isinstance(sequence_copy, PulseSequence)
    assert sequence_copy == sequence

    sequence_copy["ch3"].append(
        Pulse(
            duration=100,
            amplitude=0.9,
            envelope=Drag(rel_sigma=0.2, beta=1),
        )
    )
    assert "ch3" not in sequence
