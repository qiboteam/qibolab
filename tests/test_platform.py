"""Tests :class:`qibolab.platforms.multiqubit.MultiqubitPlatform` and
:class:`qibolab.platforms.platform.DesignPlatform`."""

import inspect
import os
import pathlib
import warnings
from pathlib import Path

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
from qibolab.kernels import Kernels
from qibolab.platform import Platform, unroll_sequences
from qibolab.platform.load import PLATFORMS
from qibolab.pulses import Delay, Gaussian, Pulse, PulseSequence, PulseType, Rectangular
from qibolab.serialize import (
    PLATFORM,
    dump_kernels,
    dump_platform,
    dump_runcard,
    load_runcard,
    load_settings,
)

from .conftest import find_instrument

nshots = 1024


def test_unroll_sequences(platform):
    qubit = next(iter(platform.qubits.values()))
    sequence = PulseSequence()
    sequence.extend(qubit.native_gates.RX.create_sequence())
    sequence[qubit.measure.name].append(Delay(duration=sequence.duration))
    sequence.extend(qubit.native_gates.MZ.create_sequence())
    total_sequence, readouts = unroll_sequences(10 * [sequence], relaxation_time=10000)
    assert len(total_sequence.ro_pulses) == 10
    assert len(readouts) == 1
    assert all(len(readouts[pulse.id]) == 10 for pulse in sequence.ro_pulses)


def test_create_platform(platform):
    assert isinstance(platform, Platform)


def test_create_platform_error():
    with pytest.raises(ValueError):
        platform = create_platform("nonexistent")


def test_create_platform_multipath(tmp_path: Path):
    some = tmp_path / "some"
    others = tmp_path / "others"
    some.mkdir()
    others.mkdir()

    for p in [
        some / "platform0",
        some / "platform1",
        others / "platform1",
        others / "platform2",
    ]:
        p.mkdir()
        (p / PLATFORM).write_text(
            inspect.cleandoc(
                f"""
                from qibolab.platform import Platform

                def create():
                    return Platform("{p.parent.name}-{p.name}", {{}}, {{}}, {{}}, {{}})
                """
            )
        )

    os.environ[PLATFORMS] = f"{some}{os.pathsep}{others}"

    def path(name):
        return tmp_path / Path(create_platform(name).name.replace("-", os.sep))

    assert path("platform0").relative_to(some)
    assert path("platform1").relative_to(some)
    assert path("platform2").relative_to(others)
    with pytest.raises(ValueError):
        create_platform("platform3")


def test_platform_sampling_rate(platform):
    assert platform.sampling_rate >= 1


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
    result = platform.execute_pulse_sequence(
        sequence, ExecutionParameters(nshots=nshots)
    )
    assert result is not None


@pytest.mark.qpu
def test_platform_execute_one_drive_pulse(qpu_platform):
    # One drive pulse
    platform = qpu_platform
    qubit = next(iter(platform.qubits.values()))
    sequence = PulseSequence()
    sequence[qubit.drive.name].append(
        Pulse(
            duration=200,
            amplitude=0.07,
            envelope=Gaussian(5),
            type=PulseType.DRIVE,
        )
    )
    result = platform.execute_pulse_sequence(
        sequence, ExecutionParameters(nshots=nshots)
    )
    assert result is not None


@pytest.mark.qpu
def test_platform_execute_one_coupler_pulse(qpu_platform):
    # One drive pulse
    platform = qpu_platform
    if len(platform.couplers) == 0:
        pytest.skip("The platform does not have couplers")
    coupler = next(iter(platform.couplers.values()))
    sequence = PulseSequence()
    sequence[coupler.flux.name].append(
        Pulse(
            duration=200,
            amplitude=0.31,
            envelope=Rectangular(),
            type=PulseType.COUPLERFLUX,
        )
    )
    result = platform.execute_pulse_sequence(
        sequence, ExecutionParameters(nshots=nshots)
    )
    assert result is not None


@pytest.mark.qpu
def test_platform_execute_one_flux_pulse(qpu_platform):
    # One flux pulse
    platform = qpu_platform
    qubit = next(iter(platform.qubits.values()))
    sequence = PulseSequence()
    sequence[qubit.flux.name].append(
        Pulse(
            duration=200,
            amplitude=0.28,
            envelope=Rectangular(),
            type=PulseType.FLUX,
        )
    )
    result = platform.execute_pulse_sequence(
        sequence, ExecutionParameters(nshots=nshots)
    )
    assert result is not None


@pytest.mark.qpu
def test_platform_execute_one_long_drive_pulse(qpu_platform):
    # Long duration
    platform = qpu_platform
    qubit = next(iter(platform.qubits.values()))
    pulse = Pulse(
        duration=8192 + 200, amplitude=0.12, envelope=Gaussian(5), type=PulseType.DRIVE
    )
    sequence = PulseSequence()
    sequence[qubit.drive.name].append(pulse)
    options = ExecutionParameters(nshots=nshots)
    if find_instrument(platform, QbloxController) is not None:
        with pytest.raises(NotImplementedError):
            platform.execute_pulse_sequence(sequence, options)
    else:
        platform.execute_pulse_sequence(sequence, options)


@pytest.mark.qpu
def test_platform_execute_one_extralong_drive_pulse(qpu_platform):
    # Extra Long duration
    platform = qpu_platform
    qubit = next(iter(platform.qubits.values()))
    pulse = Pulse(
        duration=2 * 8192 + 200,
        amplitude=0.12,
        envelope=Gaussian(5),
        type=PulseType.DRIVE,
    )
    sequence = PulseSequence()
    sequence[qubit.drive.name].append(pulse)
    options = ExecutionParameters(nshots=nshots)
    if find_instrument(platform, QbloxController) is not None:
        with pytest.raises(NotImplementedError):
            platform.execute_pulse_sequence(sequence, options)
    else:
        platform.execute_pulse_sequence(sequence, options)


@pytest.mark.qpu
def test_platform_execute_one_drive_one_readout(qpu_platform):
    """One drive pulse and one readout pulse."""
    platform = qpu_platform
    qubit_id, qubit = next(iter(platform.qubits.items()))
    sequence = PulseSequence()
    sequence.extend(platform.create_RX_pulse(qubit_id))
    sequence[qubit.measure.name].append(Delay(duration=200))
    sequence.extend(platform.create_MZ_pulse(qubit_id))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_platform_execute_multiple_drive_pulses_one_readout(qpu_platform):
    """Multiple qubit drive pulses and one readout pulse."""
    platform = qpu_platform
    qubit_id, qubit = next(iter(platform.qubits.items()))
    sequence = PulseSequence()
    sequence.extend(platform.create_RX_pulse(qubit_id))
    sequence[qubit.drive.name].append(Delay(duration=4))
    sequence.extend(platform.create_RX_pulse(qubit_id))
    sequence[qubit.drive.name].append(Delay(duration=4))
    sequence.extend(platform.create_RX_pulse(qubit_id))
    sequence[qubit.measure.name].append(Delay(duration=808))
    sequence.extend(platform.create_MZ_pulse(qubit_id))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_platform_execute_multiple_drive_pulses_one_readout_no_spacing(
    qpu_platform,
):
    """Multiple qubit drive pulses and one readout pulse with no spacing
    between them."""
    platform = qpu_platform
    qubit_id, qubit = next(iter(platform.qubits.items()))
    sequence = PulseSequence()
    sequence.extend(platform.create_RX_pulse(qubit_id))
    sequence.extend(platform.create_RX_pulse(qubit_id))
    sequence.extend(platform.create_RX_pulse(qubit_id))
    sequence[qubit.measure.name].append(Delay(duration=800))
    sequence.extend(platform.create_MZ_pulse(qubit_id))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_platform_execute_multiple_overlaping_drive_pulses_one_readout(
    qpu_platform,
):
    """Multiple overlapping qubit drive pulses and one readout pulse."""
    platform = qpu_platform
    qubit_id, qubit = next(iter(platform.qubits.items()))
    sequence = PulseSequence()
    pulse = Pulse(
        duration=200, amplitude=0.08, envelope=Gaussian(7), type=PulseType.DRIVE
    )
    sequence[qubit.drive.name].append(pulse)
    sequence[qubit.drive12.name].append(pulse.copy())
    sequence[qubit.measure.name].append(Delay(duration=800))
    sequence.extend(platform.create_MZ_pulse(qubit_id))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_platform_execute_multiple_readout_pulses(qpu_platform):
    """Multiple readout pulses."""
    platform = qpu_platform
    qubit_id, qubit = next(iter(platform.qubits.items()))
    sequence = PulseSequence()
    qd_seq1 = platform.create_RX_pulse(qubit_id)
    ro_seq1 = platform.create_MZ_pulse(qubit_id)
    qd_seq2 = platform.create_RX_pulse(qubit_id)
    ro_seq2 = platform.create_MZ_pulse(qubit_id)
    sequence.extend(qd_seq1)
    sequence[qubit.measure.name].append(Delay(duration=qd_seq1.duration))
    sequence.extend(ro_seq1)
    sequence[qubit.drive.name].append(Delay(duration=ro_seq1.duration))
    sequence.extend(qd_seq2)
    sequence[qubit.measure.name].append(Delay(duration=qd_seq2.duration))
    sequence.extend(ro_seq2)
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.skip(reason="no way of currently testing this")
@pytest.mark.qpu
@pytest.mark.xfail(
    raises=AssertionError, reason="Probabilities are not well calibrated"
)
def test_excited_state_probabilities_pulses(qpu_platform):
    platform = qpu_platform
    backend = QibolabBackend(platform)
    sequence = PulseSequence()
    for qubit_id, qubit in platform.qubits.items():
        sequence.extend(platform.create_RX_pulse(qubit_id))
        sequence[qubit.measure.name].append(Delay(duration=sequence.duration))
        sequence.extend(platform.create_MZ_pulse(qubit_id))
    result = platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=5000))

    nqubits = len(platform.qubits)
    cr = CircuitResult(backend, Circuit(nqubits), result, nshots=5000)
    probs = [
        backend.circuit_result_probabilities(cr, qubits=[qubit])
        for qubit in platform.qubits
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
    backend = QibolabBackend(platform)
    sequence = PulseSequence()
    for qubit_id, qubit in platform.qubits.items():
        if not start_zero:
            sequence[qubit.measure.name].append(
                Delay(
                    duration=platform.create_RX_pulse(qubit_id).duration,
                    channel=platform.qubits[qubit].readout.name,
                )
            )
        sequence.extend(platform.create_MZ_pulse(qubit_id))
    result = platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=5000))

    nqubits = len(platform.qubits)
    cr = CircuitResult(backend, Circuit(nqubits), result, nshots=5000)
    probs = [
        backend.circuit_result_probabilities(cr, qubits=[qubit])
        for qubit in platform.qubits
    ]
    warnings.warn(f"Ground state probabilities: {probs}")
    target_probs = np.zeros((nqubits, 2))
    target_probs[:, 0] = 1
    np.testing.assert_allclose(probs, target_probs, atol=0.05)
