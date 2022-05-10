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
                         shape=Gaussian(5)))
    seq.add(pulses.Pulse(start=60,
                         frequency=200000000.0,
                         amplitude=0.5,
                         duration=20,
                         phase=0,
                         shape=Gaussian(5)))
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
                         shape=Gaussian(5)))
    seq.add(pulses.ReadoutPulse(start=60,
                                frequency=200000000.0,
                                amplitude=0.5,
                                duration=20,
                                phase=0,
                                shape=Gaussian(5)))
    assert len(seq.pulses) == 2
    assert len(seq.qcm_pulses) == 1
    assert len(seq.qrm_pulses) == 1


class PiPulseRegression:

    def __init__(self, platform):
        if platform == "tiiq":
            self.amplitude = K.platform.pi_pulse_amplitude
            self.duration = K.platform.pi_pulse_duration
            self.frequency = K.platform.pi_pulse_frequency
            self.delay = K.platform.delay_between_pulses
            self.channel = "qcm"
        elif platform == "icarusq":
            pi_pulse = K.platform.fetch_qubit_pi_pulse(0)
            ro = K.platform.fetch_qubit_readout_pulse(0)
            self.amplitude = pi_pulse.get("amplitude")
            self.duration = pi_pulse.get("duration")
            self.frequency = pi_pulse.get("frequency")
            self.delay = 0
            self.channel = [2, 3]


class ReadoutPulseRegression:

    def __init__(self, platform):
        if platform == "tiiq":
            ro = K.platform.readout_pulse
            self.channel = "qrm"
        elif platform == "icarusq":
            ro = K.platform.fetch_qubit_readout_pulse(0)
            self.channel = [0, 1]
        self.amplitude = ro.get("amplitude")
        self.duration = ro.get("duration")
        self.frequency = ro.get("frequency")


@pytest.mark.parametrize("platform", ["tiiq", "icarusq"])
def test_pulse_sequence_add_u3(platform):
    qibo.set_backend("qibolab", platform=platform)
    seq = PulseSequence()
    seq.add_u3(0.1, 0.2, 0.3)
    assert len(seq.pulses) == 2
    
    cp = PiPulseRegression(platform)
    duration = cp.duration // 2
    np.testing.assert_allclose(seq.phase, 0.6)
    if platform == "tiiq":
        assert len(seq.qcm_pulses) == 2
        np.testing.assert_allclose(seq.time, 2 * (cp.duration // 2) + 2 * cp.delay)
    pulse1 = f"P({cp.channel}, 0, {duration}, {cp.amplitude}, {cp.frequency}, 0.3, (gaussian, 5))"
    pulse2 = f"P({cp.channel}, {duration + cp.delay}, {duration}, {cp.amplitude}, {cp.frequency}, {0.4 - np.pi}, (gaussian, 5))"
    assert seq.serial() == f"{pulse1}, {pulse2}"


@pytest.mark.parametrize("platform", ["tiiq", "icarusq"])
def test_pulse_sequence_add_two_u3(platform):
    qibo.set_backend("qibolab", platform=platform)
    seq = PulseSequence()
    seq.add_u3(0.1, 0.2, 0.3)
    seq.add_u3(0.4, 0.6, 0.5)
    assert len(seq.pulses) == 4

    cp = PiPulseRegression(platform)
    duration = cp.duration // 2
    np.testing.assert_allclose(seq.phase, 0.6 + 1.5)
    if platform == "tiiq":
        assert len(seq.qcm_pulses) == 4
        np.testing.assert_allclose(seq.time, 2 * (2 * (cp.duration // 2) + 2 * cp.delay))
    pulse1 = f"P({cp.channel}, 0, {duration}, {cp.amplitude}, {cp.frequency}, 0.3, (gaussian, 5))"
    pulse2 = f"P({cp.channel}, {duration + cp.delay}, {duration}, {cp.amplitude}, {cp.frequency}, {0.4 - np.pi}, (gaussian, 5))"
    pulse3 = f"P({cp.channel}, {2 * (duration + cp.delay)}, {duration}, {cp.amplitude}, {cp.frequency}, 1.1, (gaussian, 5))"
    pulse4 = f"P({cp.channel}, {3 * (duration + cp.delay)}, {duration}, {cp.amplitude}, {cp.frequency}, {1.5 - np.pi}, (gaussian, 5))"
    assert seq.serial() == f"{pulse1}, {pulse2}, {pulse3}, {pulse4}"


@pytest.mark.parametrize("platform", ["tiiq", "icarusq"])
def test_pulse_sequence_add_measurement(platform):
    qibo.set_backend("qibolab", platform=platform)
    seq = PulseSequence()
    seq.add_u3(0.1, 0.2, 0.3)
    seq.add_measurement()
    assert len(seq.pulses) == 3
    if platform == "tiiq":
        assert len(seq.qcm_pulses) == 2
        assert len(seq.qrm_pulses) == 1
    
    cp = PiPulseRegression(platform)
    rp = ReadoutPulseRegression(platform)
    np.testing.assert_allclose(seq.phase, 0.6)
    duration = cp.duration // 2
    pulse1 = f"P({cp.channel}, 0, {duration}, {cp.amplitude}, {cp.frequency}, 0.3, (gaussian, 5))"
    pulse2 = f"P({cp.channel}, {duration + cp.delay}, {duration}, {cp.amplitude}, {cp.frequency}, {0.4 - np.pi}, (gaussian, 5))"
    start = 2 * (duration + cp.delay) + K.platform.delay_before_readout
    pulse3 = f"P({rp.channel}, {start}, {rp.duration}, {rp.amplitude}, {rp.frequency}, {seq.phase}, rectangular)"
    assert seq.serial() == f"{pulse1}, {pulse2}, {pulse3}"


def test_hardwarecircuit_init():
    circuit = HardwareCircuit(1)
    with pytest.raises(ValueError):
        circuit = HardwareCircuit(2)


@pytest.mark.parametrize("platform", ["tiiq", "icarusq"])
def test_hardwarecircuit_create_sequence(platform):
    qibo.set_backend("qibolab", platform=platform)
    circuit = HardwareCircuit(1)
    circuit.add(gates.RX(0, theta=0.1))
    circuit.add(gates.RY(0, theta=0.2))
    with pytest.raises(RuntimeError):
        seq = circuit.create_sequence()
    circuit.add(gates.M(0))
    seq = circuit.create_sequence()
    assert len(seq) == 5
    if platform == "tiiq":
        assert len(seq.qcm_pulses) == 4
        assert len(seq.qrm_pulses) == 1

    cp = PiPulseRegression(platform)
    rp = ReadoutPulseRegression(platform)
    duration = cp.duration // 2
    phases = [np.pi / 2, 0.1 - np.pi / 2, 0.1, 0.3 - np.pi]
    for i, (pulse, phase) in enumerate(zip(seq.pulses[:-1], phases)):
        assert pulse.channel == cp.channel
        np.testing.assert_allclose(pulse.start, i * (duration + cp.delay))
        np.testing.assert_allclose(pulse.duration, duration)
        np.testing.assert_allclose(pulse.amplitude, cp.amplitude)
        np.testing.assert_allclose(pulse.frequency, cp.frequency)
        np.testing.assert_allclose(pulse.phase, phase)
    
    pulse = seq.pulses[-1]
    start = 4 * (duration + cp.delay) + K.platform.delay_before_readout
    np.testing.assert_allclose(pulse.start, start)
    np.testing.assert_allclose(pulse.duration, rp.duration)
    np.testing.assert_allclose(pulse.amplitude, rp.amplitude)
    np.testing.assert_allclose(pulse.frequency, rp.frequency)
    np.testing.assert_allclose(pulse.phase, 0.3)


@pytest.mark.parametrize("platform", ["tiiq", "icarusq"])
def test_hardwarecircuit_execute_error(platform):
    qibo.set_backend("qibolab", platform=platform)
    circuit = HardwareCircuit(1)
    circuit.add(gates.X(0))
    circuit.add(gates.M(0))
    with pytest.raises(ValueError):
        result = circuit(initial_state=np.ones(2))


@pytest.mark.xfail
def test_hardwarecircuit_execute():
    # TODO: Test this method on IcarusQ
    qibo.set_backend("qibolab", platform="tiiq")
    circuit = HardwareCircuit(1)
    circuit.add(gates.X(0))
    circuit.add(gates.M(0))
    result = circuit(nshots=100)
    # disconnect from instruments so that they are available for other tests
    K.platform.disconnect()