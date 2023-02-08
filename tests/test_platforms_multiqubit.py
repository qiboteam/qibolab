import os
import pathlib
import shutil
import warnings

import numpy as np
import pytest
import yaml
from qibo.models import Circuit
from qibo.states import CircuitResult

from qibolab.backends import QibolabBackend
from qibolab.paths import qibolab_folder
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence

qubit = 0
nshots = 1024


@pytest.fixture
def platform(platform_name):
    test_runcard = pathlib.Path(__file__) / "test_platforms_multiqubit.yml"
    original_runcard = qibolab_folder / "runcards" / f"{platform_name}.yml"
    shutil.copyfile(str(original_runcard), test_runcard)
    _platform = Platform(platform_name, test_runcard)
    _platform.connect()
    _platform.setup()
    _platform.start()
    yield _platform
    _platform.stop()
    _platform.disconnect()
    os.remove(test_runcard)


def test_abstractplatform_init(platform_name):
    with open(qibolab_folder / "runcards" / f"{platform_name}.yml") as file:
        settings = yaml.safe_load(file)
    platform = Platform(platform_name)
    assert platform.name == platform_name
    assert platform.is_connected == False
    assert len(platform.instruments) == len(settings["instruments"])
    for name in settings["instruments"]:
        assert name in platform.instruments
        assert (
            str(type(platform.instruments[name]))
            == f"<class 'qibolab.instruments.{settings['instruments'][name]['lib']}.{settings['instruments'][name]['class']}'>"
        )


def test_abstractplatform_pickle(platform_name):
    import pickle

    platform = Platform(platform_name)
    serial = pickle.dumps(platform)
    new_platform = pickle.loads(serial)
    assert new_platform.name == platform.name
    assert new_platform.runcard == platform.runcard
    assert new_platform.settings == platform.settings
    assert new_platform.is_connected == platform.is_connected


@pytest.mark.qpu
def test_abstractplatform_setup_start_stop(platform):
    pass


@pytest.mark.qpu
def test_multiqubitplatform_execute_empty(platform):
    # an empty pulse sequence
    sequence = PulseSequence()
    platform.execute_pulse_sequence(sequence, nshots)


@pytest.mark.qpu
def test_multiqubitplatform_execute_one_drive_pulse(platform):
    # One drive pulse
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    platform.execute_pulse_sequence(sequence, nshots)


@pytest.mark.qpu
def test_multiqubitplatform_execute_one_long_drive_pulse(platform):
    # Long duration
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=8192 + 200))
    with pytest.raises(NotImplementedError):
        platform.execute_pulse_sequence(sequence, nshots)


@pytest.mark.qpu
def test_multiqubitplatform_execute_one_extralong_drive_pulse(platform):
    # Extra Long duration
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=2 * 8192 + 200))
    with pytest.raises(NotImplementedError):
        platform.execute_pulse_sequence(sequence, nshots)


@pytest.mark.qpu
def test_multiqubitplatform_execute_one_drive_one_readout(platform):
    # One drive pulse and one readout pulse
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    sequence.add(platform.create_qubit_readout_pulse(qubit, start=200))
    platform.execute_pulse_sequence(sequence, nshots)


@pytest.mark.qpu
def test_multiqubitplatform_execute_multiple_drive_pulses_one_readout(platform):
    # Multiple qubit drive pulses and one readout pulse
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=204, duration=200))
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=408, duration=400))
    sequence.add(platform.create_qubit_readout_pulse(qubit, start=808))
    platform.execute_pulse_sequence(sequence, nshots)


@pytest.mark.qpu
def test_multiqubitplatform_execute_multiple_drive_pulses_one_readout_no_spacing(
    platform,
):
    # Multiple qubit drive pulses and one readout pulse with no spacing between them
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=200, duration=200))
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=400, duration=400))
    sequence.add(platform.create_qubit_readout_pulse(qubit, start=800))
    platform.execute_pulse_sequence(sequence, nshots)


@pytest.mark.qpu
def test_multiqubitplatform_execute_multiple_overlaping_drive_pulses_one_readout(
    platform,
):
    # Multiple overlapping qubit drive pulses and one readout pulse
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=200, duration=200))
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=50, duration=400))
    sequence.add(platform.create_qubit_readout_pulse(qubit, start=800))
    platform.execute_pulse_sequence(sequence, nshots)


@pytest.mark.qpu
def test_multiqubitplatform_execute_multiple_readout_pulses(platform):
    # Multiple readout pulses
    sequence = PulseSequence()
    qd_pulse1 = platform.create_qubit_drive_pulse(qubit, start=0, duration=200)
    ro_pulse1 = platform.create_qubit_readout_pulse(qubit, start=200)
    qd_pulse2 = platform.create_qubit_drive_pulse(qubit, start=(ro_pulse1.start + ro_pulse1.duration), duration=400)
    ro_pulse2 = platform.create_qubit_readout_pulse(qubit, start=(ro_pulse1.start + ro_pulse1.duration + 400))
    sequence.add(qd_pulse1)
    sequence.add(ro_pulse1)
    sequence.add(qd_pulse2)
    sequence.add(ro_pulse2)
    platform.execute_pulse_sequence(sequence, nshots)


@pytest.mark.qpu
@pytest.mark.xfail(raises=AssertionError, reason="Probabilities are not well calibrated")
def test_excited_state_probabilities_pulses(platform_name, qubit):
    backend = QibolabBackend(platform_name)
    platform = backend.platform
    qd_pulse = platform.create_RX_pulse(qubit)
    ro_pulse = platform.create_MZ_pulse(qubit, start=qd_pulse.duration)
    sequence = PulseSequence()
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)
    result = platform.execute_pulse_sequence(sequence, nshots=5000)

    cr = CircuitResult(backend, Circuit(platform.nqubits), result)
    probs = backend.circuit_result_probabilities(cr, qubits=[qubit])
    warnings.warn(f"Excited state probabilities: {probs}")
    np.testing.assert_allclose(probs, [0, 1], atol=0.05)


@pytest.mark.qpu
@pytest.mark.parametrize("start_zero", [False, True])
@pytest.mark.xfail(raises=AssertionError, reason="Probabilities are not well calibrated")
def test_ground_state_probabilities_pulses(platform_name, qubit, start_zero):
    backend = QibolabBackend(platform_name)
    platform = backend.platform
    if start_zero:
        ro_pulse = platform.create_MZ_pulse(qubit, start=0)
    else:
        qd_pulse = platform.create_RX_pulse(qubit)
        ro_pulse = platform.create_MZ_pulse(qubit, start=qd_pulse.duration)
    sequence = PulseSequence()
    sequence.add(ro_pulse)
    result = platform.execute_pulse_sequence(sequence, nshots=5000)

    cr = CircuitResult(backend, Circuit(platform.nqubits), result)
    probs = backend.circuit_result_probabilities(cr, qubits=[qubit])
    warnings.warn(f"Ground state probabilities: {probs}")
    np.testing.assert_allclose(probs, [1, 0], atol=0.05)
