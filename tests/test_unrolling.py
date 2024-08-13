"""Tests ``unrolling.py``."""

import pytest

from qibolab.pulses import Drag, Pulse, PulseSequence, Rectangular
from qibolab.unrolling import Bounds, batch


def test_bounds_update():
    ps = PulseSequence(
        [
            (
                "ch3/drive",
                Pulse(duration=40, amplitude=0.9, envelope=Drag(rel_sigma=0.2, beta=1)),
            ),
            (
                "ch2/drive",
                Pulse(duration=40, amplitude=0.9, envelope=Drag(rel_sigma=0.2, beta=1)),
            ),
            (
                "ch1/drive",
                Pulse(duration=40, amplitude=0.9, envelope=Drag(rel_sigma=0.2, beta=1)),
            ),
            (
                "ch3/probe",
                Pulse(duration=1000, amplitude=0.9, envelope=Rectangular()),
            ),
            (
                "ch2/probe",
                Pulse(duration=1000, amplitude=0.9, envelope=Rectangular()),
            ),
            (
                "ch1/probe",
                Pulse(duration=1000, amplitude=0.9, envelope=Rectangular()),
            ),
        ]
    )

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
    ps = PulseSequence(
        [
            (
                "ch3/drive",
                Pulse(duration=40, amplitude=0.9, envelope=Drag(rel_sigma=0.2, beta=1)),
            ),
            (
                "ch2/drive",
                Pulse(duration=40, amplitude=0.9, envelope=Drag(rel_sigma=0.2, beta=1)),
            ),
            (
                "ch1/drive",
                Pulse(duration=40, amplitude=0.9, envelope=Drag(rel_sigma=0.2, beta=1)),
            ),
            ("ch3/probe", Pulse(duration=1000, amplitude=0.9, envelope=Rectangular())),
            ("ch2/probe", Pulse(duration=1000, amplitude=0.9, envelope=Rectangular())),
            ("ch1/probe", Pulse(duration=1000, amplitude=0.9, envelope=Rectangular())),
        ]
    )

    sequences = 10 * [ps]

    batches = list(batch(sequences, bounds))
    assert len(batches) > 1
