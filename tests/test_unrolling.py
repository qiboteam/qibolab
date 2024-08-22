"""Tests ``unrolling.py``."""

import pytest

from qibolab.pulses import Drag, Pulse, Rectangular
from qibolab.pulses.pulse import Acquisition
from qibolab.sequence import PulseSequence
from qibolab.unrolling import Bounds, batch


def test_bounds_update():
    ps = PulseSequence.load(
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
            (
                "ch1/acquisition",
                Acquisition(duration=3000),
            ),
        ]
    )

    bounds = Bounds.update(ps)

    assert bounds.waveforms >= 40
    assert bounds.readout == 1
    assert bounds.instructions > 1


def test_bounds_add():
    bounds1 = Bounds(waveforms=2, readout=1, instructions=3)
    bounds2 = Bounds(waveforms=1, readout=2, instructions=1)

    bounds_sum = bounds1 + bounds2

    assert bounds_sum.waveforms == 3
    assert bounds_sum.readout == 3
    assert bounds_sum.instructions == 4


def test_bounds_comparison():
    bounds1 = Bounds(waveforms=2, readout=1, instructions=3)
    bounds2 = Bounds(waveforms=1, readout=2, instructions=1)

    assert bounds1 > bounds2
    assert not bounds2 < bounds1


@pytest.mark.parametrize(
    "bounds",
    [
        Bounds(waveforms=150, readout=int(10e6), instructions=int(10e6)),
        Bounds(waveforms=int(10e6), readout=10, instructions=int(10e6)),
        Bounds(waveforms=int(10e6), readout=int(10e6), instructions=20),
    ],
)
def test_batch(bounds):
    ps = PulseSequence.load(
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
            ("ch3/acquisition", Acquisition(duration=1000)),
            ("ch2/probe", Pulse(duration=1000, amplitude=0.9, envelope=Rectangular())),
            ("ch2/acquisition", Acquisition(duration=1000)),
            ("ch1/probe", Pulse(duration=1000, amplitude=0.9, envelope=Rectangular())),
            ("ch1/acquisition", Acquisition(duration=1000)),
        ]
    )

    sequences = 10 * [ps]

    batches = list(batch(sequences, bounds))
    assert len(batches) > 1
