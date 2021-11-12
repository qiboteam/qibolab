import pytest
import numpy as np
import qibo
from qibo import gates, models
from qibolab import pulses
from qibolab.circuit import PulseSequence, HardwareCircuit
# TODO: Parametrize these tests using experiment


@pytest.mark.skip
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


def test_hardwarecircuit_errors():
    qibo.set_backend("qibolab")
    c = models.Circuit(5)
    with pytest.raises(NotImplementedError):
        c._add_layer()
    with pytest.raises(NotImplementedError):
        c.fuse()


@pytest.mark.skip
def test_hardwarecircuit_sequence_duration():
    from qibolab import experiment
    qibo.set_backend("qibolab")
    c = models.Circuit(2)
    c.add(gates.RX(0, theta=0.123))
    c.add(gates.RY(0, theta=0.123))
    c.add(gates.H(0))
    c.add(gates.Align(0))
    c.add(gates.M(0))
    c.qubit_config = experiment.static.initial_calibration
    qubit_times = c._calculate_sequence_duration(c.queue) # pylint: disable=E1101
    target_qubit_times = [3.911038e-08, 0]
    np.testing.assert_allclose(qubit_times, target_qubit_times)


@pytest.mark.skip
def test_hardwarecircuit_create_pulse_sequence():
    from qibolab import experiment
    qibo.set_backend("qibolab")
    c = models.Circuit(2)
    c.add(gates.RX(0, theta=0.123))
    c.add(gates.RY(0, theta=0.123))
    c.add(gates.H(0))
    c.add(gates.Align(0))
    c.add(gates.M(0))
    c.qubit_config = experiment.static.initial_calibration
    c.qubit_config[0]["gates"]["measure"] = []
    qubit_times = np.zeros(c.nqubits) - c._calculate_sequence_duration(c.queue) # pylint: disable=E1101
    qubit_phases = np.zeros(c.nqubits)
    pulse_sequence = c.create_pulse_sequence(c.queue, qubit_times, qubit_phases) # pylint: disable=E1101
    target_pulse_sequence = "P(3, -1.940378868990046e-09, 9.70189434495023e-10, 0.375, 747382500.0, 0.0, (rectangular)), "\
                            "P(3, -9.70189434495023e-10, 9.70189434495023e-10, 0.375, 747382500.0, 90.0, (rectangular))"
    pulse_sequence.serialize() == target_pulse_sequence
