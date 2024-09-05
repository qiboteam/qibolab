from copy import deepcopy

from pydantic import TypeAdapter

from qibolab.pulses import (
    Acquisition,
    Delay,
    Drag,
    Gaussian,
    Pulse,
    Readout,
    Rectangular,
    VirtualZ,
)
from qibolab.sequence import PulseSequence


def test_init():
    sequence = PulseSequence()
    assert len(sequence) == 0


def test_init_with_iterable():
    sc = "some/probe"
    oc = "other/drive"
    c5 = "5/drive"
    seq = PulseSequence(
        [
            (sc, p)
            for p in [
                Pulse(duration=20, amplitude=0.1, envelope=Gaussian(rel_sigma=3)),
                Pulse(duration=30, amplitude=0.5, envelope=Gaussian(rel_sigma=3)),
            ]
        ]
        + [(oc, Pulse(duration=40, amplitude=0.2, envelope=Gaussian(rel_sigma=3)))]
        + [
            (c5, p)
            for p in [
                Pulse(duration=45, amplitude=1.0, envelope=Gaussian(rel_sigma=3)),
                Pulse(duration=50, amplitude=0.7, envelope=Gaussian(rel_sigma=3)),
                Pulse(duration=60, amplitude=-0.65, envelope=Gaussian(rel_sigma=3)),
            ]
        ]
    )

    assert len(seq) == 6
    assert set(seq.channels) == {sc, oc, c5}
    assert len(list(seq.channel(sc))) == 2
    assert len(list(seq.channel(oc))) == 1
    assert len(list(seq.channel(c5))) == 3


def test_serialization():
    sp = "some/probe"
    sa = "some/acquisition"
    od = "other/drive"
    of = "other/flux"

    seq = PulseSequence(
        [
            (sp, Delay(duration=10)),
            (sa, Delay(duration=20)),
            (of, Pulse(duration=10, amplitude=1, envelope=Rectangular())),
            (od, VirtualZ(phase=0.6)),
            (od, Pulse(duration=10, amplitude=1, envelope=Rectangular())),
            (sp, Pulse(duration=100, amplitude=0.3, envelope=Gaussian(rel_sigma=0.1))),
            (sa, Acquisition(duration=150)),
        ]
    )

    aslist = TypeAdapter(PulseSequence).dump_python(seq)
    assert PulseSequence.load(aslist) == seq


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
    p1 = Pulse(duration=40, amplitude=0.9, envelope=Drag(rel_sigma=0.2, beta=1))
    sequence1 = PulseSequence([("ch1", p1)])
    p2 = Pulse(duration=60, amplitude=0.9, envelope=Drag(rel_sigma=0.2, beta=1))
    sequence2 = PulseSequence([("ch2", p2)])

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

    # Check order preservation, even with various channels
    vz = VirtualZ(phase=0.1)
    s1 = PulseSequence([("a", p1), ("b", vz)])
    s2 = PulseSequence([("a", vz), ("a", p2)])
    s1.concatenate(s2)
    assert isinstance(s1[0][1], Pulse)
    assert s1[0][0] == "a"
    assert isinstance(s1[1][1], VirtualZ)
    assert s1[1][0] == "b"
    assert isinstance(s1[2][1], VirtualZ)
    assert s1[2][0] == "a"
    assert isinstance(s1[3][1], Pulse)
    assert s1[3][0] == "a"

    # Check aliases
    sa1 = deepcopy(s1)
    sc1 = deepcopy(s1)
    sa1 |= s2
    sc1.concatenate(s2)
    assert sa1 == sc1
    assert sc1 == s1 | s2


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


def test_align_to_delay():
    sequence = PulseSequence(
        [
            (
                "ch1",
                Pulse(duration=40, amplitude=0.9, envelope=Drag(rel_sigma=0.2, beta=1)),
            ),
            (
                "ch1",
                Delay(duration=20),
            ),
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
    sequence.align(["ch1", "ch2"])
    sequence.append(
        (
            "ch1",
            Pulse(duration=20, amplitude=0.1, envelope=Gaussian(rel_sigma=3)),
        )
    )
    sequence.append(
        (
            "ch2",
            Pulse(duration=30, amplitude=0.1, envelope=Gaussian(rel_sigma=3)),
        )
    )

    delay_sequence = sequence.align_to_delays()
    assert len(delay_sequence) == len(sequence) - 1
    for (ch1, p1), (ch2, p2) in zip(sequence[:5], delay_sequence[:5]):
        assert ch1 == ch2
        assert p1 is p2
    assert delay_sequence[5] == ("ch1", Delay(duration=40))
    for (ch1, p1), (ch2, p2) in zip(sequence[7:], delay_sequence[6:]):
        assert ch1 == ch2
        assert p1 is p2
    # assert that pulses after align start simultaneously
    sequence_without_last = delay_sequence[:-2]
    ch1_start = sequence_without_last.channel_duration("ch1")
    ch2_start = sequence_without_last.channel_duration("ch2")
    assert ch1_start == ch2_start


def test_trim():
    p = Pulse(duration=40, amplitude=0.9, envelope=Rectangular())
    d = Delay(duration=10)
    vz = VirtualZ(phase=0.1)
    sequence = PulseSequence(
        [
            ("a", p),
            ("a", d),
            ("b", d),
            ("b", d),
            ("c", d),
            ("c", p),
            ("d", p),
            ("d", vz),
        ]
    )
    trimmed = sequence.trim()
    # the final delay is dropped
    assert len(list(trimmed.channel("a"))) == 1
    # only delays, all dropped
    assert len(list(trimmed.channel("b"))) == 0
    # initial delay is kept
    assert len(list(trimmed.channel("c"))) == 2
    # the order is preserved
    assert isinstance(next(iter(trimmed.channel("d"))), Pulse)


def test_acquisitions():
    probe = Pulse(duration=10, amplitude=1, envelope=Rectangular())
    acq = Acquisition(duration=10)
    sequence = PulseSequence.load(
        [
            ("1/drive", VirtualZ(phase=0.7)),
            ("1/probe", Delay(duration=15)),
            ("1/acquisition", Delay(duration=20)),
            ("1/probe", probe),
            ("1/acquisition", acq),
            ("1/flux", probe),
        ]
    )
    acqs = sequence.acquisitions
    assert len(acqs) == 1
    assert acqs[0][1] is acq


def test_readouts():
    probe = Pulse(duration=10, amplitude=1, envelope=Rectangular())
    acq = Acquisition(duration=10)
    sequence = PulseSequence([("1/acquisition", Readout(probe=probe, acquisition=acq))])
    assert len(sequence) == 1
    ro = sequence[0][1]
    assert isinstance(ro, Readout)
    assert ro.duration == acq.duration
    assert ro.id == acq.id

    sequence = PulseSequence(
        [
            ("1/drive", VirtualZ(phase=0.7)),
            ("1/acquisition", Delay(duration=20)),
            ("1/acquisition", Readout(probe=probe, acquisition=acq)),
            ("1/flux", probe),
        ]
    )
    assert len(sequence) == 4
    assert len(sequence.acquisitions) == 1
    assert isinstance(sequence.acquisitions[0][1], Readout)

    aslist = TypeAdapter(PulseSequence).dump_python(sequence)
    assert PulseSequence.load(aslist) == sequence
