"""Tests ``unrolling.py``."""

import pytest

from qibolab.pulses import Drag, Pulse, PulseSequence, PulseType, Rectangular
from qibolab.unrolling import Bounds, batch


def test_bounds_update():
    p1 = Pulse(400, 40, 0.9, int(100e6), 0, Drag(5, 1), 3, PulseType.DRIVE)
    p2 = Pulse(500, 40, 0.9, int(100e6), 0, Drag(5, 1), 2, PulseType.DRIVE)
    p3 = Pulse(600, 40, 0.9, int(100e6), 0, Drag(5, 1), 1, PulseType.DRIVE)

    p4 = Pulse(440, 1000, 0.9, int(20e6), 0, Rectangular(), 3, PulseType.READOUT)
    p5 = Pulse(540, 1000, 0.9, int(20e6), 0, Rectangular(), 2, PulseType.READOUT)
    p6 = Pulse(640, 1000, 0.9, int(20e6), 0, Rectangular(), 1, PulseType.READOUT)

    ps = PulseSequence(p1, p2, p3, p4, p5, p6)
    bounds = Bounds.update(ps)

    assert bounds.waveforms == 40
    assert bounds.readout == 3
    assert bounds.instructions == 6


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
    p1 = Pulse(400, 40, 0.9, int(100e6), 0, Drag(5, 1), 3, PulseType.DRIVE)
    p2 = Pulse(500, 40, 0.9, int(100e6), 0, Drag(5, 1), 2, PulseType.DRIVE)
    p3 = Pulse(600, 40, 0.9, int(100e6), 0, Drag(5, 1), 1, PulseType.DRIVE)

    p4 = Pulse(440, 1000, 0.9, int(20e6), 0, Rectangular(), 3, PulseType.READOUT)
    p5 = Pulse(540, 1000, 0.9, int(20e6), 0, Rectangular(), 2, PulseType.READOUT)
    p6 = Pulse(640, 1000, 0.9, int(20e6), 0, Rectangular(), 1, PulseType.READOUT)

    ps = PulseSequence(p1, p2, p3, p4, p5, p6)

    sequences = 10 * [ps]

    batches = list(batch(sequences, bounds))
    assert len(batches) == 4
