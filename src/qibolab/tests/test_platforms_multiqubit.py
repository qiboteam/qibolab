from pathlib import Path
from qibolab.circuit import PulseSequence
from qibolab.paths import qibolab_folder
import yaml
import pytest
from qibolab.platforms.multiqubit import MultiqubitPlatform

hardware_available = False
platform_name = 'tiiq'
platform: MultiqubitPlatform
test_runcard: Path


@pytest.fixture()
def environment_setup():
        global test_runcard
        global platform
        original_runcard = qibolab_folder / "runcards" / f"{platform_name}.yml"
        test_runcard = qibolab_folder / "tests" / "multiqubit_test_runcard.yml"
        import shutil
        shutil.copyfile(str(original_runcard), (test_runcard))
        platform = MultiqubitPlatform(platform_name, test_runcard)

        yield
        
        import os
        os.remove(test_runcard)


def test_abstractplatform_init(environment_setup):
    with open(test_runcard, "r") as file:
        settings = yaml.safe_load(file)
    assert platform.name == platform_name
    assert platform.runcard == test_runcard
    assert platform.is_connected == False
    assert len(platform.instruments) == len(settings['instruments'])
    for name in settings['instruments']:
        assert name in platform.instruments
        assert str(type(platform.instruments[name])) == f"<class 'qibolab.instruments.{settings['instruments'][name]['lib']}.{settings['instruments'][name]['class']}'>"


def test_abstractplatform_reload_settings(environment_setup):
    original_sampling_rate = platform.settings['settings']['sampling_rate']
    new_sampling_rate = 2_000_000_000
    save_config_parameter(test_runcard, 'sampling_rate', new_sampling_rate, 'settings')
    platform.reload_settings()
    assert platform.settings['settings']['sampling_rate'] == new_sampling_rate
    save_config_parameter(test_runcard, 'sampling_rate', original_sampling_rate, 'settings')
    platform.reload_settings()
    

def save_config_parameter(runcard, parameter, value, *keys):
    with open(runcard, "r") as file:
        settings = yaml.safe_load(file)
    file.close()

    node = settings
    for key in keys:
        node = node.get(key)
    node[parameter] = value

    # store latest timestamp
    import datetime
    settings['timestamp'] = datetime.datetime.utcnow()

    with open(runcard, "w") as file:
        settings = yaml.dump(settings, file, sort_keys=False, indent=4)
    file.close()


def test_abstractplatform_pickle(environment_setup):
    import pickle
    serial = pickle.dumps(platform)
    new_platform: MultiqubitPlatform = pickle.loads(serial)
    assert new_platform.name == platform.name
    assert new_platform.runcard == platform.runcard
    assert new_platform.settings == platform.settings
    assert new_platform.is_connected == platform.is_connected


@pytest.mark.xfail
def test_abstractplatform_connect(environment_setup):
    platform.connect()
    assert platform.is_connected
    global hardware_available
    hardware_available = platform.is_connected


def test_abstractplatform_start_stop(environment_setup):
    if not hardware_available:
        pytest.xfail('Hardware not available')
    else:
        runcard = qibolab_folder / "tests" / "multiqubit_test_runcard.yml"
        platform = MultiqubitPlatform("multiqubit", runcard)
        platform.connect()
        platform.setup()
        platform.start()
        platform.stop()
        platform.disconnect()


def test_multiqubitplatform_execute_pulse_sequences(environment_setup):
    if not hardware_available:
        pytest.xfail('Hardware not available')
    else:
        from qibolab.pulses import Pulse, ReadoutPulse, Gaussian, Rectangular, Drag
        from qibolab.circuit import PulseSequence

        platform.connect()
        platform.setup()
        platform.start()

        qubit = 1 # TODO: Test all qubits
        
        qd_frequency = platform.native_gates['single_qubit'][qubit]['RX']['frequency']
        qd_amplitude = platform.native_gates['single_qubit'][qubit]['RX']['amplitude']
        qd_shape = platform.native_gates['single_qubit'][qubit]['RX']['shape']
        qd_channel = platform.qubit_channel_map[qubit][1]

        ro_frequency = platform.native_gates['single_qubit'][qubit]['MZ']['frequency']
        ro_amplitude = platform.native_gates['single_qubit'][qubit]['MZ']['amplitude']
        ro_shape = platform.native_gates['single_qubit'][qubit]['MZ']['shape']     
        ro_channel = platform.qubit_channel_map[qubit][0]
        
        phase = 0

        nshots = 1024

        qubit_drive_pulse = lambda start, duration: Pulse(start, qd_frequency, qd_amplitude, duration, phase, qd_shape, qd_channel)
        qubit_readout_pulse = lambda start, duration: ReadoutPulse(start, ro_frequency, ro_amplitude, duration, phase, ro_shape, ro_channel)
        
        # One drive pulse
        sequence = PulseSequence()
        pulse0 = qubit_drive_pulse(start = 0, duration = 200)
        sequence.add(pulse0)    
        # Short duration
        platform.execute_pulse_sequence(sequence, nshots)
        # Long duration
        pulse0.duration += 8192 
        platform.execute_pulse_sequence(sequence, nshots)
        # Extra Long duration
        pulse0.duration += 8192 
        platform.execute_pulse_sequence(sequence, nshots)

        # One drive pulse and one readout pulse
        sequence = PulseSequence()
        pulse0 = qubit_drive_pulse(start = 0, duration = 200)
        pulse1 = qubit_readout_pulse(start = 200, duration = 2000)
        sequence.add(pulse0, pulse1)  
        platform.execute_pulse_sequence(sequence, nshots)

        platform.stop()
        platform.disconnect()


@pytest.mark.xfail # not implemented
def test_multiqubitplatform_run_calibration(environment_setup):
    if not hardware_available:
        pytest.xfail('Hardware not available')
    else:
        platform.connect()
        platform.setup()
        platform.run_calibration()
