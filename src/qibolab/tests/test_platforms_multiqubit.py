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
qubit = 1
nshots = 1024


def instantiate_platform():
    global test_runcard
    global platform
    original_runcard = qibolab_folder / "runcards" / f"{platform_name}.yml"
    test_runcard = qibolab_folder / "tests" / "multiqubit_test_runcard.yml"
    import shutil
    shutil.copyfile(str(original_runcard), (test_runcard))
    platform = MultiqubitPlatform(platform_name, test_runcard)


def connect_platform():
    if hardware_available:
        platform.connect()
        platform.setup()
        platform.start()


def disconnect_platform():
    if hardware_available:
        platform.stop()
        platform.disconnect()


def cleanup():
        import os
        os.remove(test_runcard)


@pytest.fixture()
def fx_instantiate_platform():
    instantiate_platform()
    yield
    cleanup()
        

@pytest.fixture()
def fx_connect_platform():
    instantiate_platform()
    connect_platform()
    yield
    disconnect_platform()
    cleanup()


def test_abstractplatform_init(fx_instantiate_platform):
    with open(test_runcard, "r") as file:
        settings = yaml.safe_load(file)
    assert platform.name == platform_name
    assert platform.runcard == test_runcard
    assert platform.is_connected == False
    assert len(platform.instruments) == len(settings['instruments'])
    for name in settings['instruments']:
        assert name in platform.instruments
        assert str(type(platform.instruments[name])) == f"<class 'qibolab.instruments.{settings['instruments'][name]['lib']}.{settings['instruments'][name]['class']}'>"


def test_abstractplatform_reload_settings(fx_instantiate_platform):
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


def test_abstractplatform_pickle(fx_instantiate_platform):
    import pickle
    serial = pickle.dumps(platform)
    new_platform: MultiqubitPlatform = pickle.loads(serial)
    assert new_platform.name == platform.name
    assert new_platform.runcard == platform.runcard
    assert new_platform.settings == platform.settings
    assert new_platform.is_connected == platform.is_connected


@pytest.mark.xfail
def test_abstractplatform_connect_disconnect(fx_instantiate_platform):
    platform.connect()
    assert platform.is_connected
    global hardware_available
    hardware_available = platform.is_connected
    platform.disconnect()


def test_abstractplatform_setup_start_stop(fx_instantiate_platform):
    if not hardware_available:
        pytest.xfail('Hardware not available')
    else:
        platform.connect()
        platform.setup()
        platform.start()
        platform.stop()
        platform.disconnect()

def test_multiqubitplatform_execute_empty(fx_connect_platform):
    if not hardware_available:
        pytest.xfail('Hardware not available')
    else:
        # an empty pulse sequence
        sequence = PulseSequence()   
        platform.execute_pulse_sequence(sequence, nshots)


def test_multiqubitplatform_execute_one_drive_pulse(fx_connect_platform):
    if not hardware_available:
        pytest.xfail('Hardware not available')
    else:
        # One drive pulse
        sequence = PulseSequence()
        sequence.add(platform.qubit_drive_pulse(qubit, start = 0, duration = 200))    
        platform.execute_pulse_sequence(sequence, nshots)


def test_multiqubitplatform_execute_one_long_drive_pulse(fx_connect_platform):
    if not hardware_available:
        pytest.xfail('Hardware not available')
    else:
        # Long duration
        sequence = PulseSequence()
        sequence.add(platform.qubit_drive_pulse(qubit, start = 0, duration = 8192+200))  
        platform.execute_pulse_sequence(sequence, nshots)


def test_multiqubitplatform_execute_one_extralong_drive_pulse(fx_connect_platform):
    if not hardware_available:
        pytest.xfail('Hardware not available')
    else:
        # Extra Long duration
        sequence = PulseSequence()
        sequence.add(platform.qubit_drive_pulse(qubit, start = 0, duration = 2*8192+200))  
        platform.execute_pulse_sequence(sequence, nshots)


def test_multiqubitplatform_execute_one_drive_one_readout(fx_connect_platform):
    if not hardware_available:
        pytest.xfail('Hardware not available')
    else:
        # One drive pulse and one readout pulse
        sequence = PulseSequence()
        sequence.add(platform.qubit_drive_pulse(qubit, start = 0, duration = 200))
        sequence.add(platform.qubit_readout_pulse(qubit, start = 200))    
        platform.execute_pulse_sequence(sequence, nshots)

def test_multiqubitplatform_execute_multiple_drive_pulses_one_readout(fx_connect_platform):
    if not hardware_available:
        pytest.xfail('Hardware not available')
    else:
        # Multiple qubit drive pulses and one readout pulse
        sequence = PulseSequence()
        sequence.add(platform.qubit_drive_pulse(qubit, start = 0, duration = 200))
        sequence.add(platform.qubit_drive_pulse(qubit, start = 204, duration = 200)) 
        sequence.add(platform.qubit_drive_pulse(qubit, start = 408, duration = 400)) 
        sequence.add(platform.qubit_readout_pulse(qubit, start = 808)) 
        platform.execute_pulse_sequence(sequence, nshots)


def test_multiqubitplatform_execute_multiple_drive_pulses_one_readout_no_spacing(fx_connect_platform):
    if not hardware_available:
        pytest.xfail('Hardware not available')
    else:
        # Multiple qubit drive pulses and one readout pulse with no spacing between them
        sequence = PulseSequence()
        sequence.add(platform.qubit_drive_pulse(qubit, start = 0, duration = 200))
        sequence.add(platform.qubit_drive_pulse(qubit, start = 200, duration = 200)) 
        sequence.add(platform.qubit_drive_pulse(qubit, start = 400, duration = 400)) 
        sequence.add(platform.qubit_readout_pulse(qubit, start = 800)) 
        platform.execute_pulse_sequence(sequence, nshots)


@pytest.mark.xfail # not implemented yet
def test_multiqubitplatform_execute_multiple_overlaping_drive_pulses_one_readout(fx_connect_platform):
    if not hardware_available:
        pytest.xfail('Hardware not available')
    else:
        # Multiple overlapping qubit drive pulses and one readout pulse
        sequence = PulseSequence()
        sequence.add(platform.qubit_drive_pulse(qubit, start = 0, duration = 200))
        sequence.add(platform.qubit_drive_pulse(qubit, start = 200, duration = 200)) 
        sequence.add(platform.qubit_drive_pulse(qubit, start = 50, duration = 400)) 
        sequence.add(platform.qubit_readout_pulse(qubit, start = 800)) 
        platform.execute_pulse_sequence(sequence, nshots)

@pytest.mark.xfail # not implemented yet
def test_multiqubitplatform_execute_multiple_readout_pulses(fx_connect_platform):
    if not hardware_available:
        pytest.xfail('Hardware not available')
    else:
        # Multiple readout pulses
        sequence = PulseSequence()
        qd_pulse1 = platform.qubit_drive_pulse(qubit, start = 0, duration = 200)
        ro_pulse1 = platform.qubit_readout_pulse(qubit, start = 200)
        qd_pulse2 = platform.qubit_drive_pulse(qubit, start = (ro_pulse1.start + ro_pulse1.duration), duration = 400)
        ro_pulse2 = platform.qubit_readout_pulse(qubit, start = (ro_pulse1.start + ro_pulse1.duration + 400))
        sequence.add(qd_pulse1)
        sequence.add(ro_pulse1)
        sequence.add(qd_pulse2)
        sequence.add(ro_pulse2)

        platform.execute_pulse_sequence(sequence, nshots)


@pytest.mark.xfail # not implemented yet
def test_multiqubitplatform_run_calibration(environment_setup):
    if not hardware_available:
        pytest.xfail('Hardware not available')
    else:
        platform.connect()
        platform.setup()
        platform.run_calibration()
