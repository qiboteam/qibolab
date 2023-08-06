"""Tests :class:`qibolab.platforms.multiqubit.MultiqubitPlatform` and
:class:`qibolab.platforms.platform.DesignPlatform`.
"""
import pickle
import warnings

import numpy as np
import pytest
from qibo.models import Circuit
from qibo.states import CircuitResult

from qibolab import create_platform
from qibolab.backends import QibolabBackend
from qibolab.execution_parameters import ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence

nshots = 1024


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


# TODO: this test should be improved
@pytest.mark.parametrize(
    "par",
    [
        "readout_frequency",
        "sweetspot",
        "threshold",
        "bare_resonator_frequency",
        "drive_frequency",
        "iq_angle",
        "mean_gnd_states",
        "mean_exc_states",
        "classifiers_hpars",
    ],
)
def test_update(platform, par):
    qubits = {q: qubit for q, qubit in platform.qubits.items() if qubit.readout is not None and qubit.drive is not None}
    new_values = np.ones(len(qubits))
    if "states" in par:
        updates = {par: {q: [new_values[i], new_values[i]] for i, q in enumerate(qubits)}}
    else:
        updates = {par: {q: new_values[i] for i, q in enumerate(qubits)}}
    platform.update(updates)
    for i, qubit in qubits.items():
        value = updates[par][i]
        if "frequency" in par:
            value *= 1e9
        if "states" in par:
            assert value == getattr(qubit, par)
        else:
            assert value == float(getattr(qubit, par))


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
@pytest.mark.xfail(raises=NotImplementedError, reason="Qblox does not support long waveforms.")
def test_platform_execute_one_long_drive_pulse(qpu_platform):
    # Long duration
    platform = qpu_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=8192 + 200))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
@pytest.mark.xfail(raises=NotImplementedError, reason="Qblox does not support long waveforms.")
def test_platform_execute_one_extralong_drive_pulse(qpu_platform):
    # Extra Long duration
    platform = qpu_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=2 * 8192 + 200))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


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
