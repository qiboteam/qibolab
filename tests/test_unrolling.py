"""Tests ``unrolling.py``."""

from qibolab._core.platform import Platform
from qibolab._core.pulses import Delay, Drag, Pulse, Rectangular
from qibolab._core.pulses.pulse import Acquisition
from qibolab._core.sequence import PulseSequence
from qibolab._core.unrolling import Bounds, unroll_sequences


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


def test_unroll_sequences(platform: Platform):
    qubit = next(iter(platform.qubits.values()))
    assert qubit.probe is not None
    natives = platform.natives.single_qubit[0]
    assert natives.RX is not None
    assert natives.MZ is not None
    sequence = PulseSequence()
    sequence.concatenate(natives.RX.create_sequence())
    sequence.append((qubit.probe, Delay(duration=sequence.duration)))
    sequence.concatenate(natives.MZ.create_sequence())
    total_sequence, readouts = unroll_sequences(10 * [sequence], relaxation_time=10000)
    assert len(total_sequence.acquisitions) == 10
    assert len(readouts) == 1
    assert all(len(readouts[acq.id]) == 10 for _, acq in sequence.acquisitions)
