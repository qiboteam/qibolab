import pytest
import numpy as np
import qibo
from qibo import K, gates
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
    # TODO: Test this method on IcarusQ (requires qubit)
    qibo.set_backend("qibolab")
    seq = PulseSequence()
    seq.add_u3(0.1, 0.2, 0.3)
    seq.add_u3(0.4, 0.6, 0.5)
    assert len(seq.pulses) == 4
    assert len(seq.qcm_pulses) == 4
    np.testing.assert_allclose(seq.phase, 1.1)

    amplitude = K.platform.pi_pulse_amplitude
    duration = K.platform.pi_pulse_duration
    frequency = K.platform.pi_pulse_frequency
    delay = K.platform.delay_between_pulses
    np.testing.assert_allclose(seq.time, 2 * (duration + delay))

    duration = duration // 2
    pulse1 = f"P(qcm, 0, {duration}, {amplitude}, {frequency}, -1.3707963267948966, gaussian({duration / 5}))"
    pulse2 = f"P(qcm, {duration + delay}, {duration}, {amplitude}, {frequency}, 1.6707963267948964, gaussian({duration / 5}))"
    pulse3 = f"P(qcm, {2 * (duration + delay)}, {duration}, {amplitude}, {frequency}, -0.5707963267948967, gaussian({duration / 5}))"
    pulse4 = f"P(qcm, {3 * (duration + delay)}, {duration}, {amplitude}, {frequency}, 2.1707963267948966, gaussian({duration / 5}))"
    assert seq.serial() == f"{pulse1}, {pulse2}, {pulse3}, {pulse4}"


def test_pulse_sequence_add_measurement():
    # TODO: Test this method on IcarusQ (requires qubit)
    qibo.set_backend("qibolab")
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
    pulse3 = f"P(qrm, {start}, {ro.get('duration')}, {ro.get('amplitude')}, {ro.get('frequency')}, {seq.phase}, rectangular)"
    assert seq.serial() == f"{pulse1}, {pulse2}, {pulse3}"


def test_hardwarecircuit_init():
    circuit = HardwareCircuit(1)
    with pytest.raises(ValueError):
        circuit = HardwareCircuit(2)


def test_hardwarecircuit_create_sequence():
    # TODO: Test this method on IcarusQ (requires qubit)
    qibo.set_backend("qibolab")
    circuit = HardwareCircuit(1)
    circuit.add(gates.RX(0, theta=0.1))
    circuit.add(gates.RY(0, theta=0.2))
    with pytest.raises(RuntimeError):
        seq = circuit.create_sequence()
    circuit.add(gates.M(0))
    seq = circuit.create_sequence()
    
    assert len(seq) == 5
    assert len(seq.qcm_pulses) == 4
    assert len(seq.qrm_pulses) == 1

    amplitude = K.platform.pi_pulse_amplitude
    duration = K.platform.pi_pulse_duration
    frequency = K.platform.pi_pulse_frequency
    delay = K.platform.delay_between_pulses
    duration = duration // 2
    std = duration / 5
    pulse1 = f"P(qcm, 0, {duration}, {amplitude}, {frequency}, -3.141592653589793, gaussian({std}))"
    pulse2 = f"P(qcm, {duration + delay}, {duration}, {amplitude}, {frequency}, -0.10000000000000009, gaussian({std}))"
    pulse3 = f"P(qcm, {2 * (duration + delay)}, {duration}, {amplitude}, {frequency}, -1.6707963267948966, gaussian({std}))"
    pulse4 = f"P(qcm, {3 * (duration + delay)}, {duration}, {amplitude}, {frequency}, 1.2707963267948963, gaussian({std}))"
    ro = K.platform.readout_pulse
    pulse5 = f"P(qrm, 652, {ro.get('duration')}, {ro.get('amplitude')}, {ro.get('frequency')}, -0.30000000000000027, rectangular)"
    assert seq.serial() == ", ".join([pulse1, pulse2, pulse3, pulse4, pulse5])


def test_hardwarecircuit_execute():
    # TODO: Test this method on IcarusQ (requires qubit)
    qibo.set_backend("qibolab")
    circuit = HardwareCircuit(1)
    circuit.add(gates.X(0))
    circuit.add(gates.M(0))
    with pytest.raises(ValueError):
        result = circuit(initial_state=np.ones(2))

# TODO: Test ``HardwareCircuit.execute``