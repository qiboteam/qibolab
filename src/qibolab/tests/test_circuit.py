import pytest
import numpy as np
import qibo
from qibo import K, gates
from qibolab.pulses import Pulse, ReadoutPulse
from qibolab.circuit import PulseSequence, HardwareCircuit
from qibolab.paths import qibolab_folder
from qibolab.platforms.multiqubit import MultiqubitPlatform


qubit = 1


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


class PiPulseRegression:

    def __init__(self, platform_name, qubit):
        original_runcard = qibolab_folder / "runcards" / f"{platform_name}.yml"
        test_runcard = qibolab_folder / "tests" / "multiqubit_test_runcard.yml"
        import shutil
        shutil.copyfile(str(original_runcard), (test_runcard))
        platform = MultiqubitPlatform(platform_name, test_runcard)

        self.duration = platform.settings['native_gates']['single_qubit'][qubit]['RX']['duration']
        self.amplitude = platform.settings['native_gates']['single_qubit'][qubit]['RX']['amplitude']
        self.frequency = platform.settings['native_gates']['single_qubit'][qubit]['RX']['frequency']
        self.phase = 0
        self.shape = platform.settings['native_gates']['single_qubit'][qubit]['RX']['shape']
        self.channel = platform.settings['qubit_channel_map'][qubit][1]


class ReadoutPulseRegression:

    def __init__(self, platform_name, qubit):
        original_runcard = qibolab_folder / "runcards" / f"{platform_name}.yml"
        test_runcard = qibolab_folder / "tests" / "multiqubit_test_runcard.yml"
        import shutil
        shutil.copyfile(str(original_runcard), (test_runcard))
        platform = MultiqubitPlatform(platform_name, test_runcard)

        self.duration = platform.settings['native_gates']['single_qubit'][qubit]['MZ']['duration']
        self.amplitude = platform.settings['native_gates']['single_qubit'][qubit]['MZ']['amplitude']
        self.frequency = platform.settings['native_gates']['single_qubit'][qubit]['MZ']['frequency']
        self.phase = 0
        self.shape = platform.settings['native_gates']['single_qubit'][qubit]['MZ']['shape']
        self.channel = platform.settings['qubit_channel_map'][qubit][0]


@pytest.mark.parametrize("platform_name", ['tiiq', 'qili', 'multiqubit']) #, 'icarusq'])
def test_pulse_sequence_add_u3(platform_name):
    qibo.set_backend("qibolab", platform=platform_name)
    seq = PulseSequence()
    K.platform.add_u3_to_pulse_sequence(seq, 0.1, 0.2, 0.3, qubit)
    assert len(seq.pulses) == 2
    assert len(seq.qd_pulses) == 2
    
    cp = PiPulseRegression(platform_name, qubit)
    duration = cp.duration
    np.testing.assert_allclose(seq.time, 2 * duration)
    np.testing.assert_allclose(seq.phase, 0.6)

    pulse1 = f"Pulse({0}, {cp.duration}, {cp.amplitude/2}, {cp.frequency}, 0.3, '{cp.shape}', {cp.channel}, 'qd')"
    pulse2 = f"Pulse({cp.duration}, {cp.duration}, {cp.amplitude/2}, {cp.frequency}, {0.4 - np.pi}, '{cp.shape}', {cp.channel}, 'qd')"

    assert seq.serial == f"{pulse1}, {pulse2}"


@pytest.mark.parametrize("platform_name", ['tiiq', 'qili', 'multiqubit']) #, 'icarusq'])
def test_pulse_sequence_add_two_u3(platform_name):
    qibo.set_backend("qibolab", platform=platform_name)
    seq = PulseSequence()
    K.platform.add_u3_to_pulse_sequence(seq, 0.1, 0.2, 0.3, qubit)
    K.platform.add_u3_to_pulse_sequence(seq, 0.4, 0.6, 0.5, qubit)
    assert len(seq.pulses) == 4
    assert len(seq.qd_pulses) == 4

    cp = PiPulseRegression(platform_name, qubit)
    duration = cp.duration
    np.testing.assert_allclose(seq.phase, 0.6 + 1.5)
    np.testing.assert_allclose(seq.time, 2 * 2 * cp.duration)

    pulse1 = f"Pulse({0}, {cp.duration}, {cp.amplitude/2}, {cp.frequency}, 0.3, '{cp.shape}', {cp.channel}, 'qd')"
    pulse2 = f"Pulse({cp.duration}, {cp.duration}, {cp.amplitude/2}, {cp.frequency}, {0.4 - np.pi}, '{cp.shape}', {cp.channel}, 'qd')"
    pulse3 = f"Pulse({2 * cp.duration}, {cp.duration}, {cp.amplitude/2}, {cp.frequency}, 1.1, '{cp.shape}', {cp.channel}, 'qd')"
    pulse4 = f"Pulse({3 * cp.duration}, {cp.duration}, {cp.amplitude/2}, {cp.frequency}, {1.5 - np.pi}, '{cp.shape}', {cp.channel}, 'qd')"

    assert seq.serial == f"{pulse1}, {pulse2}, {pulse3}, {pulse4}"


@pytest.mark.parametrize("platform_name", ['tiiq', 'qili', 'multiqubit']) #, 'icarusq'])
def test_pulse_sequence_add_measurement(platform_name):
    qibo.set_backend("qibolab", platform=platform_name)
    seq = PulseSequence()
    K.platform.add_u3_to_pulse_sequence(seq, 0.1, 0.2, 0.3, qubit)
    K.platform.add_measurement_to_pulse_sequence(seq, qubit)
    assert len(seq.pulses) == 3
    assert len(seq.qd_pulses) == 2
    assert len(seq.ro_pulses) == 1
    
    cp = PiPulseRegression(platform_name, qubit)
    ro = ReadoutPulseRegression(platform_name, qubit)
    np.testing.assert_allclose(seq.phase, 0.6)


    pulse1 = f"Pulse({0}, {cp.duration}, {cp.amplitude/2}, {cp.frequency}, 0.3, '{cp.shape}', {cp.channel}, 'qd')"
    pulse2 = f"Pulse({cp.duration}, {cp.duration}, {cp.amplitude/2}, {cp.frequency}, {0.4 - np.pi}, '{cp.shape}', {cp.channel}, 'qd')"
    pulse3 = f"ReadoutPulse({2 * cp.duration}, {ro.duration}, {ro.amplitude}, {ro.frequency}, {seq.phase}, '{ro.shape}', {ro.channel}, 'ro')"
    assert seq.serial == f"{pulse1}, {pulse2}, {pulse3}"


@pytest.mark.xfail # to implement fetching the nqubits from the platform
def test_hardwarecircuit_init():
    circuit = HardwareCircuit(1)
    with pytest.raises(ValueError):
        circuit = HardwareCircuit(2)

@pytest.mark.xfail # to move circuit.create_sequence logit to AbstractPlatform
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

    cp = PiPulseRegression(platform_name, qubit)
    rp = ReadoutPulseRegression(platform_name, qubit)
    
    phases = [np.pi / 2, 0.1 - np.pi / 2, 0.1, 0.3 - np.pi]
    for i, (pulse, phase) in enumerate(zip(seq.pulses[:-1], phases)):
        assert pulse.channel == cp.channel
        np.testing.assert_allclose(pulse.start, i * cp.duration)
        np.testing.assert_allclose(pulse.duration, cp.duration)
        np.testing.assert_allclose(pulse.amplitude, cp.amplitude/2)
        np.testing.assert_allclose(pulse.frequency, cp.frequency)
        np.testing.assert_allclose(pulse.phase, phase)
    
    pulse = seq.pulses[-1]
    start = 4 * cp.duration
    np.testing.assert_allclose(pulse.start, start)
    np.testing.assert_allclose(pulse.duration, rp.duration)
    np.testing.assert_allclose(pulse.amplitude, rp.amplitude)
    np.testing.assert_allclose(pulse.frequency, rp.frequency)
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