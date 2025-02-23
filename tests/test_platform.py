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
from qibolab._core.backends import QibolabBackend
from qibolab._core.components import AcquisitionConfig, IqConfig, OscillatorConfig
from qibolab._core.dummy import create_dummy
from qibolab._core.dummy.platform import FOLDER
from qibolab._core.native import SingleQubitNatives, TwoQubitNatives
from qibolab._core.parameters import NativeGates, Parameters, update_configs
from qibolab._core.platform import Platform
from qibolab._core.platform.load import PLATFORM, PLATFORMS, locate_platform
from qibolab._core.platform.platform import PARAMETERS
from qibolab._core.pulses import Delay, Gaussian, Pulse, Rectangular
from qibolab._core.sequence import PulseSequence
from qibolab._core.serialize import replace

nshots = 1024


def test_create_platform(platform):
    assert isinstance(platform, Platform)


def test_create_platform_error():
    with pytest.raises(ValueError):
        _ = create_platform("nonexistent")


def test_create_platform_from_hardware():
    original_value = os.environ.get(PLATFORMS)
    os.environ[PLATFORMS] = str(Path(__file__).parent)
    _ = create_platform("dummy_hardware")
    os.remove(Path(__file__).parent / "dummy_hardware" / PARAMETERS)
    os.environ[PLATFORMS] = original_value


def test_platform_basics():
    platform = Platform(
        name="ciao",
        parameters=Parameters(native_gates=NativeGates()),
        instruments={},
        qubits={},
    )
    assert str(platform) == "ciao"
    assert platform.pairs == []

    qs = {q: SingleQubitNatives() for q in range(10)}
    ts = {(q1, q2): TwoQubitNatives() for q1 in range(3) for q2 in range(4, 8)}
    platform2 = Platform(
        name="come va?",
        parameters=Parameters(
            native_gates=NativeGates(
                single_qubit=qs,
                two_qubit=ts,
                coupler={},
            )
        ),
        instruments={},
        qubits=qs,
    )
    assert str(platform2) == "come va?"
    assert (1, 6) in platform2.pairs


def test_locate_platform(tmp_path: Path):
    some = tmp_path / "some"
    some.mkdir()

    for p in [some / "platform0", some / "platform1"]:
        p.mkdir()
        (p / PLATFORM).write_text("'Ciao'")

    assert locate_platform("platform0", [some]) == some / "platform0"

    with pytest.raises(ValueError):
        locate_platform("platform3")

    os.environ[PLATFORMS] = str(some)

    assert locate_platform("platform1") == some / "platform1"


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
                from qibolab._core.platform import Platform

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


def test_duplicated_acquisition():
    """A shallow copy will duplicate the object.

    This leads to non-unique identifiers across all sequences, and it's
    then flagged as an error (since unique identifiers are assumed in
    the return type, to avoid overwriting dict entries).
    """
    platform = create_platform("dummy")
    sequence = platform.natives.single_qubit[0].MZ.create_sequence()

    with pytest.raises(ValueError, match="unique"):
        _ = platform.execute([sequence, sequence.copy()])


def test_update_configs(platform):
    drive_name = "q0/drive"
    pump_name = "twpa_pump"
    configs = {
        drive_name: IqConfig(frequency=4.1e9),
        pump_name: OscillatorConfig(frequency=3e9, power=-5),
    }

    updated = update_configs(configs, [{drive_name: {"frequency": 4.2e9}}])
    assert updated is None
    assert configs[drive_name].frequency == 4.2e9

    update_configs(
        configs, [{drive_name: {"frequency": 4.3e9}, pump_name: {"power": -10}}]
    )
    assert configs[drive_name].frequency == 4.3e9
    assert configs[pump_name].frequency == 3e9
    assert configs[pump_name].power == -10

    update_configs(
        configs,
        [{drive_name: {"frequency": 4.4e9}}, {drive_name: {"frequency": 4.5e9}}],
    )
    assert configs[drive_name].frequency == 4.5e9

    with pytest.raises(ValueError, match="unknown component"):
        update_configs(configs, [{"non existent": {"property": 1.0}}])


def test_dump_parameters(platform: Platform, tmp_path: Path):
    (tmp_path / PARAMETERS).write_text(platform.parameters.model_dump_json())
    final = Parameters.model_validate_json((tmp_path / PARAMETERS).read_text())
    if platform.name == "dummy":
        target = Parameters.model_validate_json((FOLDER / PARAMETERS).read_text())
    else:
        target_path = pathlib.Path(__file__).parent / "dummy_qrc" / f"{platform.name}"
        target = Parameters.model_validate_json((target_path / PARAMETERS).read_text())

    # assert configs section is dumped properly in the parameters
    assert final.configs == target.configs


def test_dump_parameters_with_updates(platform: Platform, tmp_path: Path):
    qubit = next(iter(platform.qubits.values()))
    frequency = platform.config(qubit.drive).frequency + 1.5e9
    smearing = platform.config(qubit.acquisition).smearing + 10
    update = {
        str(qubit.drive): {"frequency": frequency},
        str(qubit.acquisition): {"smearing": smearing},
    }
    update_configs(platform.parameters.configs, [update])
    (tmp_path / PARAMETERS).write_text(platform.parameters.model_dump_json())
    final = Parameters.model_validate_json((tmp_path / PARAMETERS).read_text())
    assert final.configs[qubit.drive].frequency == frequency
    assert final.configs[qubit.acquisition].smearing == smearing


def test_kernels(tmp_path: Path):
    """Test dumping and loading of `Kernels`."""

    platform = create_dummy()
    for name, config in platform.parameters.configs.items():
        if isinstance(config, AcquisitionConfig):
            platform.parameters.configs[name] = replace(
                config, kernel=np.random.rand(10)
            )

    platform.dump(tmp_path)
    reloaded = Platform.load(
        tmp_path,
        instruments=platform.instruments,
        qubits=platform.qubits,
        couplers=platform.couplers,
    )

    for qubit in platform.qubits.values():
        orig = platform.parameters.configs[qubit.acquisition].kernel
        load = reloaded.parameters.configs[qubit.acquisition].kernel
        np.testing.assert_array_equal(orig, load)


def test_dump_platform(tmp_path):
    """Test platform dump and loading parameters and kernels."""

    platform = create_dummy()

    platform.dump(tmp_path)

    settings = Parameters.model_validate_json(
        (tmp_path / PARAMETERS).read_text()
    ).settings

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
    result = platform.execute([sequence], nshots=nshots)
    assert result is not None


@pytest.mark.qpu
def test_platform_execute_one_drive_pulse(qpu_platform):
    # One drive pulse
    platform = qpu_platform
    qubit = next(iter(platform.qubits.values()))
    sequence = PulseSequence(
        [
            (
                qubit.drive,
                Pulse(duration=200, amplitude=0.07, envelope=Gaussian(0.2)),
            )
        ]
    )
    result = platform.execute([sequence], nshots=nshots)
    assert result is not None


@pytest.mark.qpu
def test_platform_execute_one_coupler_pulse(qpu_platform):
    # One drive pulse
    platform = qpu_platform
    if len(platform.couplers) == 0:
        pytest.skip("The platform does not have couplers")
    coupler = next(iter(platform.couplers.values()))
    sequence = PulseSequence(
        [
            (
                coupler.flux,
                Pulse(duration=200, amplitude=0.31, envelope=Rectangular()),
            )
        ]
    )
    result = platform.execute([sequence], nshots=nshots)
    assert result is not None


@pytest.mark.qpu
def test_platform_execute_one_flux_pulse(qpu_platform):
    # One flux pulse
    platform = qpu_platform
    qubit = next(iter(platform.qubits.values()))
    sequence = PulseSequence(
        [
            (
                qubit.flux,
                Pulse(duration=200, amplitude=0.28, envelope=Rectangular()),
            )
        ]
    )
    result = platform.execute([sequence], nshots=nshots)
    assert result is not None


@pytest.mark.qpu
def test_platform_execute_one_long_drive_pulse(qpu_platform):
    # Long duration
    platform = qpu_platform
    qubit = next(iter(platform.qubits.values()))
    pulse = Pulse(duration=8192 + 200, amplitude=0.12, envelope=Gaussian(5))
    sequence = PulseSequence([(qubit.drive, pulse)])
    platform.execute([sequence], nshots=nshots)


@pytest.mark.qpu
def test_platform_execute_one_extralong_drive_pulse(qpu_platform):
    # Extra Long duration
    platform = qpu_platform
    qubit = next(iter(platform.qubits.values()))
    pulse = Pulse(duration=2 * 8192 + 200, amplitude=0.12, envelope=Gaussian(0.2))
    sequence = PulseSequence([(qubit.drive, pulse)])
    platform.execute([sequence], nshots=nshots)


@pytest.mark.qpu
def test_platform_execute_one_drive_one_readout(qpu_platform):
    """One drive pulse and one readout pulse."""
    platform = qpu_platform
    qubit_id, qubit = next(iter(platform.qubits.items()))
    sequence = PulseSequence()
    sequence.concatenate(platform.create_RX_pulse(qubit_id))
    sequence.append((qubit.probe, Delay(duration=200)))
    sequence.concatenate(platform.create_MZ_pulse(qubit_id))
    platform.execute([sequence], nshots=nshots)


@pytest.mark.qpu
def test_platform_execute_multiple_drive_pulses_one_readout(qpu_platform):
    """Multiple qubit drive pulses and one readout pulse."""
    platform = qpu_platform
    qubit_id, qubit = next(iter(platform.qubits.items()))
    sequence = PulseSequence()
    sequence.concatenate(platform.create_RX_pulse(qubit_id))
    sequence.append((qubit.drive, Delay(duration=4)))
    sequence.concatenate(platform.create_RX_pulse(qubit_id))
    sequence.append((qubit.drive, Delay(duration=4)))
    sequence.concatenate(platform.create_RX_pulse(qubit_id))
    sequence.append((qubit.probe, Delay(duration=808)))
    sequence.concatenate(platform.create_MZ_pulse(qubit_id))
    platform.execute([sequence], nshots=nshots)


@pytest.mark.qpu
def test_platform_execute_multiple_drive_pulses_one_readout_no_spacing(
    qpu_platform,
):
    """Multiple qubit drive pulses and one readout pulse with no spacing
    between them."""
    platform = qpu_platform
    qubit_id, qubit = next(iter(platform.qubits.items()))
    sequence = PulseSequence()
    sequence.concatenate(platform.create_RX_pulse(qubit_id))
    sequence.concatenate(platform.create_RX_pulse(qubit_id))
    sequence.concatenate(platform.create_RX_pulse(qubit_id))
    sequence.append((qubit.probe, Delay(duration=800)))
    sequence.concatenate(platform.create_MZ_pulse(qubit_id))
    platform.execute([sequence], nshots=nshots)


@pytest.mark.qpu
def test_platform_execute_multiple_overlaping_drive_pulses_one_readout(
    qpu_platform,
):
    """Multiple overlapping qubit drive pulses and one readout pulse."""
    platform = qpu_platform
    qubit_id, qubit = next(iter(platform.qubits.items()))
    pulse = Pulse(duration=200, amplitude=0.08, envelope=Gaussian(rel_sigma=1 / 7))
    sequence = PulseSequence(
        [
            (qubit.drive, pulse),
            (qubit.drive12, pulse.model_copy()),
            (qubit.probe, Delay(duration=800)),
        ]
    )
    sequence.concatenate(platform.create_MZ_pulse(qubit_id))
    platform.execute([sequence], nshots=nshots)


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
    sequence.concatenate(qd_seq1)
    sequence.append((qubit.probe, Delay(duration=qd_seq1.duration)))
    sequence.concatenate(ro_seq1)
    sequence.append((qubit.drive, Delay(duration=ro_seq1.duration)))
    sequence.concatenate(qd_seq2)
    sequence.append((qubit.probe, Delay(duration=qd_seq2.duration)))
    sequence.concatenate(ro_seq2)
    platform.execute([sequence], nshots=nshots)


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
        sequence.concatenate(platform.create_RX_pulse(qubit_id))
        sequence.append((qubit.probe, Delay(duration=sequence.duration)))
        sequence.concatenate(platform.create_MZ_pulse(qubit_id))
    result = platform.execute([sequence], nshots=5000)

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
            sequence.append(
                (
                    qubit.probe,
                    Delay(
                        duration=platform.create_RX_pulse(qubit_id).duration,
                    ),
                )
            )
        sequence.concatenate(platform.create_MZ_pulse(qubit_id))
    result = platform.execute([sequence], nshots=5000)

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
