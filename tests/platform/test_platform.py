import pathlib
from pathlib import Path

import numpy as np
import pytest

from qibolab._core.components import AcquisitionConfig, IqConfig, OscillatorConfig
from qibolab._core.dummy import create_dummy
from qibolab._core.dummy.platform import FOLDER as DUMMY_FOLDER
from qibolab._core.native import SingleQubitNatives, TwoQubitNatives
from qibolab._core.parameters import NativeGates, Parameters, update_configs
from qibolab._core.platform import Platform
from qibolab._core.platform.platform import PARAMETERS
from qibolab._core.pulses import Delay
from qibolab._core.sequence import PulseSequence
from qibolab._core.serialize import replace
from qibolab.platform import create_platform


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


def test_unknown_channel_raises_helpful_error():
    platform = create_platform("dummy")
    sequence = PulseSequence([("missing/channel", Delay(duration=10))])

    with pytest.raises(
        ValueError, match=r"Unknown channel\(s\) in pulse sequence: missing/channel"
    ):
        platform.execute([sequence])


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
        target = Parameters.model_validate_json((DUMMY_FOLDER / PARAMETERS).read_text())
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
