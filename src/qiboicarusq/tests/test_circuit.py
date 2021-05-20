import pytest
import numpy as np
import qiboicarusq
from qiboicarusq.circuit import PulseSequence, HardwareCircuit
# TODO: Parametrize these tests using experiment


def test_pulse_sequence_init():
    seq = PulseSequence([])
    assert seq.pulses == []
    assert seq.duration == 1.391304347826087e-05
    assert seq.sample_size == 32000
    seq = PulseSequence([], duration=2e-6)
    assert seq.pulses == []
    assert seq.duration == 2e-6
    assert seq.sample_size == 4600
