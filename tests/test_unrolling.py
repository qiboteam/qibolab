"""Tests ``unrolling.py``."""

from qibolab._core.platform import Platform
from qibolab._core.pulses import Delay
from qibolab._core.sequence import PulseSequence
from qibolab._core.unrolling import unroll_sequences


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
