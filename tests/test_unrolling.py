"""Tests ``unrolling.py``."""

import pytest

from qibolab.pulses import Drag, Pulse, PulseSequence, PulseType, Rectangular
from qibolab.unrolling import Bounds, batch


def test_bounds_update():
    p1 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=int(100e6),
        envelope=Drag(rel_sigma=0.2, beta=1),
        channel="3",
        type=PulseType.DRIVE,
    )
    p2 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=int(100e6),
        envelope=Drag(rel_sigma=0.2, beta=1),
        channel="2",
        type=PulseType.DRIVE,
    )
    p3 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=int(100e6),
        envelope=Drag(rel_sigma=0.2, beta=1),
        channel="1",
        type=PulseType.DRIVE,
    )

    p4 = Pulse(
        duration=1000,
        amplitude=0.9,
        frequency=int(20e6),
        envelope=Rectangular(),
        channel="3",
        type=PulseType.READOUT,
    )
    p5 = Pulse(
        duration=1000,
        amplitude=0.9,
        frequency=int(20e6),
        envelope=Rectangular(),
        channel="2",
        type=PulseType.READOUT,
    )
    p6 = Pulse(
        duration=1000,
        amplitude=0.9,
        frequency=int(20e6),
        envelope=Rectangular(),
        channel="1",
        type=PulseType.READOUT,
    )

    ps = PulseSequence([p1, p2, p3, p4, p5, p6])
    bounds = Bounds.update(ps)

    assert bounds.waveforms >= 40
    assert bounds.readout == 3
    assert bounds.instructions > 1


def test_bounds_add():
    bounds1 = Bounds(2, 1, 3)
    bounds2 = Bounds(1, 2, 1)

    bounds_sum = bounds1 + bounds2

    assert bounds_sum.waveforms == 3
    assert bounds_sum.readout == 3
    assert bounds_sum.instructions == 4


def test_bounds_comparison():
    bounds1 = Bounds(2, 1, 3)
    bounds2 = Bounds(1, 2, 1)

    assert bounds1 > bounds2
    assert not bounds2 < bounds1


@pytest.mark.parametrize(
    "bounds",
    [
        Bounds(150, int(10e6), int(10e6)),
        Bounds(int(10e6), 10, int(10e6)),
        Bounds(int(10e6), int(10e6), 20),
    ],
)
def test_batch(bounds):
    p1 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=int(100e6),
        envelope=Drag(rel_sigma=0.2, beta=1),
        channel="3",
        type=PulseType.DRIVE,
    )
    p2 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=int(100e6),
        envelope=Drag(rel_sigma=0.2, beta=1),
        channel="2",
        type=PulseType.DRIVE,
    )
    p3 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=int(100e6),
        envelope=Drag(rel_sigma=0.2, beta=1),
        channel="1",
        type=PulseType.DRIVE,
    )

    p4 = Pulse(
        duration=1000,
        amplitude=0.9,
        frequency=int(20e6),
        envelope=Rectangular(),
        channel="3",
        type=PulseType.READOUT,
    )
    p5 = Pulse(
        duration=1000,
        amplitude=0.9,
        frequency=int(20e6),
        envelope=Rectangular(),
        channel="2",
        type=PulseType.READOUT,
    )
    p6 = Pulse(
        duration=1000,
        amplitude=0.9,
        frequency=int(20e6),
        envelope=Rectangular(),
        channel="1",
        type=PulseType.READOUT,
    )

    ps = PulseSequence([p1, p2, p3, p4, p5, p6])

    sequences = 10 * [ps]

    batches = list(batch(sequences, bounds))
    assert len(batches) > 1
