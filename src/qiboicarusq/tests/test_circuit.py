import pytest
import numpy as np
from qiboicarusq import pulses
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


@pytest.mark.skip("Skipping this test because `seq.file_dir` is not available")
def test_pulse_sequence_compile():
    seq = PulseSequence([
        pulses.BasicPulse(0, 0.5, 1.5, 0.8, 40.00, 0.7, pulses.Gaussian(1.0)),
        pulses.FilePulse(0, 1.0, "file"),
        pulses.BasicPulse(0, 0.5, 1.5, 0.8, 40.00, 0.7, pulses.Drag(1.0, 1.5))
        ])
    waveform = seq.compile()
    target_waveform = np.zeros_like(waveform)
    np.testing.assert_allclose(waveform, target_waveform)


def test_pulse_sequence_serialize():
    seq = PulseSequence([
        pulses.BasicPulse(0, 0.5, 1.5, 0.8, 40.00, 0.7, pulses.Gaussian(1.0)),
        pulses.FilePulse(0, 1.0, "file"),
        pulses.BasicPulse(0, 0.5, 1.5, 0.8, 40.00, 0.7, pulses.Drag(1.0, 1.5))
        ])
    target_repr = "P(0, 0.5, 1.5, 0.8, 40.0, 0.7, (gaussian, 1.0)), "\
                  "F(0, 1.0, file), "\
                  "P(0, 0.5, 1.5, 0.8, 40.0, 0.7, (drag, 1.0, 1.5))"
    assert seq.serialize() == target_repr


# TODO: Test HardwareCircuit
