"""Tests :class:`qibolab.platforms.multiqubit.MultiqubitPlatform` and
:class:`qibolab.platforms.platform.DesignPlatform`."""
import pathlib
import pickle
import warnings

import numpy as np
import pytest
from qibo.models import Circuit
from qibo.result import CircuitResult

from qibolab import create_platform
from qibolab.backends import QibolabBackend
from qibolab.dummy import create_dummy
from qibolab.dummy.platform import FOLDER
from qibolab.execution_parameters import ExecutionParameters
from qibolab.instruments.qblox.controller import QbloxController
from qibolab.instruments.rfsoc.driver import RFSoC
from qibolab.kernels import Kernels
from qibolab.platform import Platform, unroll_sequences
from qibolab.pulses import PulseSequence, Rectangular
from qibolab.serialize import (
    dump_kernels,
    dump_platform,
    dump_runcard,
    load_runcard,
    load_settings,
)

from .conftest import find_instrument

nshots = 1024


def test_unroll_sequences(platform):
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    qd_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_MZ_pulse(qubit, start=qd_pulse.finish)
    sequence.append(qd_pulse)
    sequence.append(ro_pulse)
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


def test_platform_sampling_rate(platform):
    assert platform.sampling_rate >= 1


@pytest.mark.xfail(reason="Cannot pickle all platforms")
def test_platform_pickle(platform):
    serial = pickle.dumps(platform)
    new_platform = pickle.loads(serial)
    assert new_platform.name == platform.name
    assert new_platform.is_connected == platform.is_connected


def test_dump_runcard(platform, tmp_path):
    dump_runcard(platform, tmp_path)
    final_runcard = load_runcard(tmp_path)
    if platform.name == "dummy" or platform.name == "dummy_couplers":
        target_runcard = load_runcard(FOLDER)
    else:
        target_path = pathlib.Path(__file__).parent / "dummy_qrc" / f"{platform.name}"
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


@pytest.mark.parametrize("has_kernels", [False, True])
def test_kernels(tmp_path, has_kernels):
    """Test dumping and loading of `Kernels`."""

    platform = create_dummy()
    if has_kernels:
        for qubit in platform.qubits:
            platform.qubits[qubit].kernel = np.random.rand(10)

    dump_kernels(platform, tmp_path)

    if has_kernels:
        kernels = Kernels.load(tmp_path)
        for qubit in platform.qubits:
            np.testing.assert_array_equal(platform.qubits[qubit].kernel, kernels[qubit])
    else:
        with pytest.raises(FileNotFoundError):
            Kernels.load(tmp_path)


@pytest.mark.parametrize("has_kernels", [False, True])
def test_dump_platform(tmp_path, has_kernels):
    """Test platform dump and loading runcard and kernels."""

    platform = create_dummy()
    if has_kernels:
        for qubit in platform.qubits:
            platform.qubits[qubit].kernel = np.random.rand(10)

    dump_platform(platform, tmp_path)

    settings = load_settings(load_runcard(tmp_path))
    if has_kernels:
        kernels = Kernels.load(tmp_path)
        for qubit in platform.qubits:
            np.testing.assert_array_equal(platform.qubits[qubit].kernel, kernels[qubit])

    assert settings == platform.settings


@pytest.fixture(scope="module")
def qpu_platform(connected_platform):
    connected_platform.connect()
    yield connected_platform


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
    sequence.append(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_platform_execute_one_coupler_pulse(qpu_platform):
    # One drive pulse
    platform = qpu_platform
    if len(platform.couplers) == 0:
        pytest.skip("The platform does not have couplers")
    coupler = next(iter(platform.couplers))
    sequence = PulseSequence()
    sequence.append(
        platform.create_coupler_pulse(coupler, start=0, duration=200, amplitude=1)
    )
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))
    assert len(sequence.cf_pulses) > 0


@pytest.mark.qpu
def test_platform_execute_one_long_drive_pulse(qpu_platform):
    # Long duration
    platform = qpu_platform
    qubit = next(iter(platform.qubits))
    pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=8192 + 200)
    sequence = PulseSequence()
    sequence.append(pulse)
    options = ExecutionParameters(nshots=nshots)
    if find_instrument(platform, QbloxController) is not None:
        with pytest.raises(NotImplementedError):
            platform.execute_pulse_sequence(sequence, options)
    elif find_instrument(platform, RFSoC) is not None and not isinstance(
        pulse.shape, Rectangular
    ):
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
    sequence.append(pulse)
    options = ExecutionParameters(nshots=nshots)
    if find_instrument(platform, QbloxController) is not None:
        with pytest.raises(NotImplementedError):
            platform.execute_pulse_sequence(sequence, options)
    elif find_instrument(platform, RFSoC) is not None and not isinstance(
        pulse.shape, Rectangular
    ):
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
    sequence.append(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    sequence.append(platform.create_qubit_readout_pulse(qubit, start=200))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_platform_execute_multiple_drive_pulses_one_readout(qpu_platform):
    # Multiple qubit drive pulses and one readout pulse
    platform = qpu_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    sequence.append(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    sequence.append(platform.create_qubit_drive_pulse(qubit, start=204, duration=200))
    sequence.append(platform.create_qubit_drive_pulse(qubit, start=408, duration=400))
    sequence.append(platform.create_qubit_readout_pulse(qubit, start=808))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_platform_execute_multiple_drive_pulses_one_readout_no_spacing(
    qpu_platform,
):
    # Multiple qubit drive pulses and one readout pulse with no spacing between them
    platform = qpu_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    sequence.append(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    sequence.append(platform.create_qubit_drive_pulse(qubit, start=200, duration=200))
    sequence.append(platform.create_qubit_drive_pulse(qubit, start=400, duration=400))
    sequence.append(platform.create_qubit_readout_pulse(qubit, start=800))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_platform_execute_multiple_overlaping_drive_pulses_one_readout(
    qpu_platform,
):
    # Multiple overlapping qubit drive pulses and one readout pulse
    platform = qpu_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    sequence.append(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    sequence.append(platform.create_qubit_drive_pulse(qubit, start=200, duration=200))
    sequence.append(platform.create_qubit_drive_pulse(qubit, start=50, duration=400))
    sequence.append(platform.create_qubit_readout_pulse(qubit, start=800))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_platform_execute_multiple_readout_pulses(qpu_platform):
    # Multiple readout pulses
    platform = qpu_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    qd_pulse1 = platform.create_qubit_drive_pulse(qubit, start=0, duration=200)
    ro_pulse1 = platform.create_qubit_readout_pulse(qubit, start=200)
    qd_pulse2 = platform.create_qubit_drive_pulse(
        qubit, start=(ro_pulse1.start + ro_pulse1.duration), duration=400
    )
    ro_pulse2 = platform.create_qubit_readout_pulse(
        qubit, start=(ro_pulse1.start + ro_pulse1.duration + 400)
    )
    sequence.append(qd_pulse1)
    sequence.append(ro_pulse1)
    sequence.append(qd_pulse2)
    sequence.append(ro_pulse2)
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.skip(reason="no way of currently testing this")
@pytest.mark.qpu
@pytest.mark.xfail(
    raises=AssertionError, reason="Probabilities are not well calibrated"
)
def test_excited_state_probabilities_pulses(qpu_platform):
    platform = qpu_platform
    qubits = [q for q, qb in platform.qubits.items() if qb.drive is not None]
    backend = QibolabBackend(platform)
    sequence = PulseSequence()
    for qubit in qubits:
        qd_pulse = platform.create_RX_pulse(qubit)
        ro_pulse = platform.create_MZ_pulse(qubit, start=qd_pulse.duration)
        sequence.append(qd_pulse)
        sequence.append(ro_pulse)
    result = platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=5000))

    nqubits = len(qubits)
    cr = CircuitResult(backend, Circuit(nqubits), result, nshots=5000)
    probs = [
        backend.circuit_result_probabilities(cr, qubits=[qubit]) for qubit in qubits
    ]
    warnings.warn(f"Excited state probabilities: {probs}")
    target_probs = np.zeros((nqubits, 2))
    target_probs[:, 1] = 1
    np.testing.assert_allclose(probs, target_probs, atol=0.05)


@pytest.mark.skip(reason="no way of currently testing this")
@pytest.mark.qpu
@pytest.mark.parametrize("start_zero", [False, True])
@pytest.mark.xfail(
    raises=AssertionError, reason="Probabilities are not well calibrated"
)
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
        sequence.append(ro_pulse)
    result = platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=5000))

    nqubits = len(qubits)
    cr = CircuitResult(backend, Circuit(nqubits), result, nshots=5000)
    probs = [
        backend.circuit_result_probabilities(cr, qubits=[qubit]) for qubit in qubits
    ]
    warnings.warn(f"Ground state probabilities: {probs}")
    target_probs = np.zeros((nqubits, 2))
    target_probs[:, 0] = 1
    np.testing.assert_allclose(probs, target_probs, atol=0.05)
