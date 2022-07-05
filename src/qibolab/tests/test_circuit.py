import pytest
import numpy as np
import qibo
from qibo import K, gates
from qibolab.pulses import Pulse, ReadoutPulse
from qibolab.circuit import PulseSequence, HardwareCircuit
from qibolab.paths import qibolab_folder
from qibolab.platforms.multiqubit import MultiqubitPlatform


qubit = 0


def test_pulse_sequence_add():
    sequence = PulseSequence()
    sequence.add(Pulse(start=0,
                    frequency=200_000_000,
                    amplitude=0.3,
                    duration=60,
                    phase=0,
                    shape='Gaussian(5)',
                    channel=1)) 
    sequence.add(Pulse(start=64,
                    frequency=200_000_000,
                    amplitude=0.3,
                    duration=30,
                    phase=0,
                    shape='Gaussian(5)',
                    channel=1)) 
    assert len(sequence.pulses) == 2
    assert len(sequence.qd_pulses) == 2


def test_pulse_sequence_add_readout():
    sequence = PulseSequence()
    sequence.add(Pulse(start=0,
                    frequency=200_000_000,
                    amplitude=0.3,
                    duration=60,
                    phase=0,
                    shape='Gaussian(5)',
                    channel=1)) 

    sequence.add(Pulse(start=64,
                frequency=200_000_000,
                amplitude=0.3,
                duration=60,
                phase=0,
                shape='Drag(5, 2)', 
                channel=1,
                type = 'qf')) 

    sequence.add(ReadoutPulse(start=128,
                        frequency=20_000_000,
                        amplitude=0.9,
                        duration=2000,
                        phase=0,
                        shape='Rectangular()', 
                        channel=11)) 
    assert len(sequence.pulses) == 3
    assert len(sequence.ro_pulses) == 1
    assert len(sequence.qd_pulses) == 1
    assert len(sequence.qf_pulses) == 1


@pytest.mark.parametrize("platform_name", ['tiiq', 'qili', 'multiqubit']) #, 'icarusq'])
def test_pulse_sequence_add_u3(platform_name):
    qibo.set_backend("qibolab", platform=platform_name)
    seq = PulseSequence()
    seq.add_u3(0.1, 0.2, 0.3, qubit)
    assert len(seq.pulses) == 2
    assert len(seq.qd_pulses) == 2
    
    RX90_pulse1 = K.platform.RX90_pulse(qubit, start = 0, phase = 0.3)
    RX90_pulse2 = K.platform.RX90_pulse(qubit, start = (RX90_pulse1.start + RX90_pulse1.duration), phase = 0.4 - np.pi)

    np.testing.assert_allclose(seq.time, RX90_pulse1.duration + RX90_pulse2.duration)
    np.testing.assert_allclose(seq.phase, 0.6)

    assert seq.serial == f"{RX90_pulse1.serial}, {RX90_pulse2.serial}"


@pytest.mark.parametrize("platform_name", ['tiiq', 'qili', 'multiqubit']) #, 'icarusq'])
def test_pulse_sequence_add_two_u3(platform_name):
    qibo.set_backend("qibolab", platform=platform_name)
    seq = PulseSequence()
    seq.add_u3(0.1, 0.2, 0.3, qubit)
    seq.add_u3(0.4, 0.6, 0.5, qubit)
    assert len(seq.pulses) == 4
    assert len(seq.qd_pulses) == 4

    RX90_pulse = K.platform.RX90_pulse(qubit)
    np.testing.assert_allclose(seq.phase, 0.6 + 1.5)
    np.testing.assert_allclose(seq.time, 2 * 2 * RX90_pulse.duration)

    RX90_pulse1 = K.platform.RX90_pulse(qubit, start = 0, phase = 0.3)
    RX90_pulse2 = K.platform.RX90_pulse(qubit, start = (RX90_pulse1.start + RX90_pulse1.duration), phase = 0.4 - np.pi)
    RX90_pulse3 = K.platform.RX90_pulse(qubit, start = (RX90_pulse2.start + RX90_pulse2.duration), phase = 1.1)
    RX90_pulse4 = K.platform.RX90_pulse(qubit, start = (RX90_pulse3.start + RX90_pulse3.duration), phase = 1.5 - np.pi)

    assert seq.serial == f"{RX90_pulse1.serial}, {RX90_pulse2.serial}, {RX90_pulse3.serial}, {RX90_pulse4.serial}"


@pytest.mark.parametrize("platform_name", ['tiiq', 'qili', 'multiqubit']) #, 'icarusq'])
def test_pulse_sequence_add_measurement(platform_name):
    qibo.set_backend("qibolab", platform=platform_name)
    seq = PulseSequence()
    seq.add_u3(0.1, 0.2, 0.3, qubit)
    seq.add_measurement(qubit)
    assert len(seq.pulses) == 3
    assert len(seq.qd_pulses) == 2
    assert len(seq.ro_pulses) == 1
    
    np.testing.assert_allclose(seq.phase, 0.6)
    
    RX90_pulse1 = K.platform.RX90_pulse(qubit, start = 0, phase = 0.3)
    RX90_pulse2 = K.platform.RX90_pulse(qubit, start = RX90_pulse1.duration, phase = 0.4 - np.pi)
    MZ_pulse = K.platform.MZ_pulse(qubit, start = (RX90_pulse2.start + RX90_pulse2.duration), phase = 0.6)
    assert seq.serial == f"{RX90_pulse1.serial}, {RX90_pulse2.serial}, {MZ_pulse.serial}"


@pytest.mark.xfail # to implement fetching the number of qubits from the platform
def test_hardwarecircuit_init():
    circuit = HardwareCircuit(1)
    with pytest.raises(ValueError):
        circuit = HardwareCircuit(2)


@pytest.mark.parametrize("platform_name", ['tiiq', 'qili', 'multiqubit']) #, 'icarusq'])
def test_hardwarecircuit_create_sequence(platform_name):
    qibo.set_backend("qibolab", platform=platform_name)
    circuit = HardwareCircuit(1)
    circuit.add(gates.RX(0, theta=0.1))
    circuit.add(gates.RY(0, theta=0.2))
    with pytest.raises(RuntimeError):
        seq = circuit.create_sequence()
    circuit.add(gates.M(0))
    seq = circuit.create_sequence()
    assert len(seq.pulses) == 5
    assert len(seq.qd_pulses) == 4
    assert len(seq.ro_pulses) == 1

    RX_pulse = K.platform.RX_pulse(qubit)
    MZ_pulse = K.platform.MZ_pulse(qubit, RX_pulse.duration)
    
    phases = [np.pi / 2, 0.1 - np.pi / 2, 0.1, 0.3 - np.pi]
    for i, (pulse, phase) in enumerate(zip(seq.pulses[:-1], phases)):
        assert pulse.channel == RX_pulse.channel
        np.testing.assert_allclose(pulse.start, i * RX_pulse.duration)
        np.testing.assert_allclose(pulse.duration, RX_pulse.duration)
        np.testing.assert_allclose(pulse.amplitude, RX_pulse.amplitude/2)
        np.testing.assert_allclose(pulse.frequency, RX_pulse.frequency)
        np.testing.assert_allclose(pulse.phase, phase)
    
    pulse = seq.pulses[-1]
    start = 4 * RX_pulse.duration
    np.testing.assert_allclose(pulse.start, start)
    np.testing.assert_allclose(pulse.duration, MZ_pulse.duration)
    np.testing.assert_allclose(pulse.amplitude, MZ_pulse.amplitude)
    np.testing.assert_allclose(pulse.frequency, MZ_pulse.frequency)
    np.testing.assert_allclose(pulse.phase, 0.3)


@pytest.mark.parametrize("platform", ['tiiq', 'qili', 'multiqubit']) #, 'icarusq'])
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