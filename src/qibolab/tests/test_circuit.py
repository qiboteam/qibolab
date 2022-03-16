import pytest
import numpy as np
import qibo
from qibo import K, gates, models
from qibolab import pulses
from qibolab.circuit import PulseSequence, HardwareCircuit


def test_pulse_sequence_add():
    from qibolab.pulse_shapes import Gaussian
    seq = PulseSequence()
    seq.add(pulses.Pulse(start=0,
                         frequency=200000000.0,
                         amplitude=0.3,
                         duration=60,
                         phase=0,
                         shape=Gaussian(60 / 5)))
    seq.add(pulses.Pulse(start=60,
                         frequency=200000000.0,
                         amplitude=0.5,
                         duration=20,
                         phase=0,
                         shape=Gaussian(20 / 5)))
    assert len(seq.pulses) == 2
    assert len(seq.qcm_pulses) == 2


def test_pulse_sequence_add_readout():
    from qibolab.pulse_shapes import Gaussian
    seq = PulseSequence()
    seq.add(pulses.Pulse(start=0,
                         frequency=200000000.0,
                         amplitude=0.3,
                         duration=60,
                         phase=0,
                         shape=Gaussian(60 / 5)))
    seq.add(pulses.ReadoutPulse(start=60,
                                frequency=200000000.0,
                                amplitude=0.5,
                                duration=20,
                                phase=0,
                                shape=Gaussian(20 / 5)))
    assert len(seq.pulses) == 2
    assert len(seq.qcm_pulses) == 1
    assert len(seq.qrm_pulses) == 1


def test_pulse_sequence_add_u3():
    # TODO: Test this method on IcarusQ (requires qubit)
    qibo.set_backend("qibolab")
    seq = PulseSequence()
    seq.add_u3(0.1, 0.2, 0.3)
    assert len(seq.pulses) == 2
    assert len(seq.qcm_pulses) == 2
    np.testing.assert_allclose(seq.phase, 0.4)

    amplitude = K.platform.pi_pulse_amplitude
    duration = K.platform.pi_pulse_duration
    frequency = K.platform.pi_pulse_frequency
    delay = K.platform.delay_between_pulses
    np.testing.assert_allclose(seq.time, duration + delay)
    duration = duration // 2
    pulse1 = f"P(qcm, 0, {duration}, {amplitude}, {frequency}, -1.3707963267948966, gaussian({duration / 5}))"
    pulse2 = f"P(qcm, {duration + delay}, {duration}, {amplitude}, {frequency}, 1.6707963267948964, gaussian({duration / 5}))"
    assert seq.serial() == f"{pulse1}, {pulse2}"


def test_pulse_sequence_add_two_u3():
    # TODO: Test adding two U3 gates
    # TODO: Test this method on IcarusQ (requires qubit)
    pass


def test_pulse_sequence_add_measurement():
    # TODO: Test this method on IcarusQ (requires qubit)
    seq = PulseSequence()
    seq.add_u3(0.1, 0.2, 0.3)
    seq.add_measurement()
    assert len(seq.pulses) == 3
    assert len(seq.qcm_pulses) == 2
    assert len(seq.qrm_pulses) == 1
    np.testing.assert_allclose(seq.phase, 0.4)

    amplitude = K.platform.pi_pulse_amplitude
    duration = K.platform.pi_pulse_duration
    frequency = K.platform.pi_pulse_frequency
    delay = K.platform.delay_between_pulses
    duration = duration // 2
    pulse1 = f"P(qcm, 0, {duration}, {amplitude}, {frequency}, -1.3707963267948966, gaussian({duration / 5}))"
    pulse2 = f"P(qcm, {duration + delay}, {duration}, {amplitude}, {frequency}, 1.6707963267948964, gaussian({duration / 5}))"
    start = 2 * (duration + delay) + K.platform.delay_before_readout
    ro = K.platform.readout_pulse
    readout = f"P(qrm, {start}, {ro.get('duration')}, {ro.get('amplitude')}, {ro.get('frequency')}, {seq.phase}, rectangular)"
    assert seq.serial() == f"{pulse1}, {pulse2}, {readout}"


@pytest.mark.skip
def test_pulse_sequence_compile():
    seq = PulseSequence([
        pulses.Pulse(0.5, 1.5, 0.8, 40.00, 0.7, Gaussian(1.0)),
        pulses.FilePulse(0, 1.0, "file"),
        pulses.Pulse(0.5, 1.5, 0.8, 40.00, 0.7, Drag(1.0, 1.5))
        ])
    waveform = seq.compile()
    target_waveform = np.zeros_like(waveform)
    np.testing.assert_allclose(waveform, target_waveform)


@pytest.mark.skip
def test_pulse_sequence_serialize():
    seq = PulseSequence([
        pulses.Pulse(0.5, 1.5, 0.8, 40.00, 0.7, Gaussian(1.0)),
        pulses.FilePulse(0, 1.0, "file"),
        pulses.Pulse(0.5, 1.5, 0.8, 40.00, 0.7, Drag(1.0, 1.5))
        ])
    target_repr = "P(0, 0.5, 1.5, 0.8, 40.0, 0.7, (gaussian, 1.0)), "\
                  "F(0, 1.0, file), "\
                  "P(0, 0.5, 1.5, 0.8, 40.0, 0.7, (drag, 1.0, 1.5))"
    assert seq.serialize() == target_repr


@pytest.mark.skip
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
