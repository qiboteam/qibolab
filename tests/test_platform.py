"""Tests :class:`qibolab.platforms.multiqubit.MultiqubitPlatform` and
:class:`qibolab.platforms.platform.DesignPlatform`.
"""
import os
import pathlib
import pickle
import warnings

import numpy as np
import pytest
from qibo.models import Circuit

from qibolab import create_platform
from qibolab.backends import QibolabBackend
from qibolab.execution_parameters import ExecutionParameters
from qibolab.instruments.qblox.controller import QbloxController
from qibolab.instruments.rfsoc.driver import RFSoC
from qibolab.platform import Platform, unroll_sequences
from qibolab.pulses import PulseSequence, Rectangular
from qibolab.serialize import dump_runcard, load_runcard

from .conftest import find_instrument

nshots = 1024


def test_unroll_sequences(platform):
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    qd_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_MZ_pulse(qubit, start=qd_pulse.finish)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)
    total_sequence, readouts = unroll_sequences(10 * [sequence], relaxation_time=10000)
    assert len(total_sequence) == 20
    assert len(total_sequence.ro_pulses) == 10
    assert total_sequence.finish == 10 * sequence.finish + 90000
    assert len(readouts) == 1
    assert len(readouts[ro_pulse.serial]) == 10


def test_create_platform(platform):
    assert isinstance(platform, Platform)


def test_create_platform_error():
    with pytest.raises(ValueError):
        platform = create_platform("nonexistent")


def test_platform_pickle(platform):
    serial = pickle.dumps(platform)
    new_platform = pickle.loads(serial)
    assert new_platform.name == platform.name
    assert new_platform.is_connected == platform.is_connected


def test_dump_runcard(platform):
    path = pathlib.Path(__file__).parent / "test.yml"
    dump_runcard(platform, path)
    final_runcard = load_runcard(path)
    target_path = pathlib.Path(__file__).parent / "dummy_qrc" / f"{platform.name}.yml"
    target_runcard = load_runcard(target_path)
    # for the characterization section the dumped runcard may contain
    # some default ``Qubit`` parameters
    target_char = target_runcard.pop("characterization")["single_qubit"]
    final_char = final_runcard.pop("characterization")["single_qubit"]
    assert final_runcard == target_runcard
    for qubit, values in target_char.items():
        for name, value in values.items():
            assert final_char[qubit][name] == value
    # assert instrument section is dumped properly in the runcard
    target_instruments = target_runcard.pop("instruments")
    final_instruments = final_runcard.pop("instruments")
    assert final_instruments == target_instruments
    os.remove(path)


@pytest.fixture(scope="module")
def qpu_platform(connected_platform):
    connected_platform.connect()
    connected_platform.setup()
    connected_platform.start()
    yield connected_platform
    connected_platform.stop()


@pytest.mark.qpu
def test_platform_setup_start_stop(qpu_platform):
    pass


@pytest.mark.qpu
def test_platform_execute_empty(qpu_platform):
    # an empty pulse sequence
    platform = qpu_platform
    sequence = PulseSequence()
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_platform_execute_one_drive_pulse(qpu_platform):
    # One drive pulse
    platform = qpu_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_platform_execute_one_coupler_pulse(qpu_platform):
    # One drive pulse
    platform = qpu_platform
    coupler = next(iter(platform.couplers))
    sequence = PulseSequence()
    sequence.add(platform.create_coupler_pulse(coupler, start=0, duration=200, amplitude=1))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))
    assert len(sequence.cf_pulses) > 0


@pytest.mark.qpu
def test_platform_execute_one_long_drive_pulse(qpu_platform):
    # Long duration
    platform = qpu_platform
    qubit = next(iter(platform.qubits))
    pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=8192 + 200)
    sequence = PulseSequence()
    sequence.add(pulse)
    options = ExecutionParameters(nshots=nshots)
    if find_instrument(platform, QbloxController) is not None:
        with pytest.raises(NotImplementedError):
            platform.execute_pulse_sequence(sequence, options)
    elif find_instrument(platform, RFSoC) is not None and not isinstance(pulse.shape, Rectangular):
        with pytest.raises(RuntimeError):
            platform.execute_pulse_sequence(sequence, options)
    else:
        platform.execute_pulse_sequence(sequence, options)


@pytest.mark.qpu
def test_platform_execute_one_extralong_drive_pulse(qpu_platform):
    # Extra Long duration
    platform = qpu_platform
    qubit = next(iter(platform.qubits))
    pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=2 * 8192 + 200)
    sequence = PulseSequence()
    sequence.add(pulse)
    options = ExecutionParameters(nshots=nshots)
    if find_instrument(platform, QbloxController) is not None:
        with pytest.raises(NotImplementedError):
            platform.execute_pulse_sequence(sequence, options)
    elif find_instrument(platform, RFSoC) is not None and not isinstance(pulse.shape, Rectangular):
        with pytest.raises(RuntimeError):
            platform.execute_pulse_sequence(sequence, options)
    else:
        platform.execute_pulse_sequence(sequence, options)


@pytest.mark.qpu
def test_platform_execute_one_drive_one_readout(qpu_platform):
    # One drive pulse and one readout pulse
    platform = qpu_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    sequence.add(platform.create_qubit_readout_pulse(qubit, start=200))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_platform_execute_multiple_drive_pulses_one_readout(qpu_platform):
    # Multiple qubit drive pulses and one readout pulse
    platform = qpu_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=204, duration=200))
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=408, duration=400))
    sequence.add(platform.create_qubit_readout_pulse(qubit, start=808))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_platform_execute_multiple_drive_pulses_one_readout_no_spacing(
    qpu_platform,
):
    # Multiple qubit drive pulses and one readout pulse with no spacing between them
    platform = qpu_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=200, duration=200))
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=400, duration=400))
    sequence.add(platform.create_qubit_readout_pulse(qubit, start=800))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_platform_execute_multiple_overlaping_drive_pulses_one_readout(
    qpu_platform,
):
    # Multiple overlapping qubit drive pulses and one readout pulse
    platform = qpu_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=200, duration=200))
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=50, duration=400))
    sequence.add(platform.create_qubit_readout_pulse(qubit, start=800))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_platform_execute_multiple_readout_pulses(qpu_platform):
    # Multiple readout pulses
    platform = qpu_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    qd_pulse1 = platform.create_qubit_drive_pulse(qubit, start=0, duration=200)
    ro_pulse1 = platform.create_qubit_readout_pulse(qubit, start=200)
    qd_pulse2 = platform.create_qubit_drive_pulse(qubit, start=(ro_pulse1.start + ro_pulse1.duration), duration=400)
    ro_pulse2 = platform.create_qubit_readout_pulse(qubit, start=(ro_pulse1.start + ro_pulse1.duration + 400))
    sequence.add(qd_pulse1)
    sequence.add(ro_pulse1)
    sequence.add(qd_pulse2)
    sequence.add(ro_pulse2)
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.skip(reason="no way of currently testing this")
@pytest.mark.qpu
@pytest.mark.xfail(raises=AssertionError, reason="Probabilities are not well calibrated")
def test_excited_state_probabilities_pulses(qpu_platform):
    platform = qpu_platform
    qubits = [q for q, qb in platform.qubits.items() if qb.drive is not None]
    backend = QibolabBackend(platform)
    sequence = PulseSequence()
    for qubit in qubits:
        qd_pulse = platform.create_RX_pulse(qubit)
        ro_pulse = platform.create_MZ_pulse(qubit, start=qd_pulse.duration)
        sequence.add(qd_pulse)
        sequence.add(ro_pulse)
    result = platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=5000))

    nqubits = len(qubits)
    cr = CircuitResult(backend, Circuit(nqubits), result, nshots=5000)
    probs = [backend.circuit_result_probabilities(cr, qubits=[qubit]) for qubit in qubits]
    warnings.warn(f"Excited state probabilities: {probs}")
    target_probs = np.zeros((nqubits, 2))
    target_probs[:, 1] = 1
    np.testing.assert_allclose(probs, target_probs, atol=0.05)


@pytest.mark.skip(reason="no way of currently testing this")
@pytest.mark.qpu
@pytest.mark.parametrize("start_zero", [False, True])
@pytest.mark.xfail(raises=AssertionError, reason="Probabilities are not well calibrated")
def test_ground_state_probabilities_pulses(qpu_platform, start_zero):
    platform = qpu_platform
    qubits = [q for q, qb in platform.qubits.items() if qb.drive is not None]
    backend = QibolabBackend(platform)
    sequence = PulseSequence()
    for qubit in qubits:
        if start_zero:
            ro_pulse = platform.create_MZ_pulse(qubit, start=0)
        else:
            qd_pulse = platform.create_RX_pulse(qubit)
            ro_pulse = platform.create_MZ_pulse(qubit, start=qd_pulse.duration)
        sequence.add(ro_pulse)
    result = platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=5000))

    nqubits = len(qubits)
    cr = CircuitResult(backend, Circuit(nqubits), result, nshots=5000)
    probs = [backend.circuit_result_probabilities(cr, qubits=[qubit]) for qubit in qubits]
    warnings.warn(f"Ground state probabilities: {probs}")
    target_probs = np.zeros((nqubits, 2))
    target_probs[:, 0] = 1
    np.testing.assert_allclose(probs, target_probs, atol=0.05)
