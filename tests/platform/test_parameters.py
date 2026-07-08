import pytest

from qibolab._core.components.channels import IqChannel
from qibolab._core.components.configs import IqConfig, IqMixerConfig, OscillatorConfig
from qibolab._core.platform.load import create_platform, locate_platform
from qibolab._core.platform.parameters import reset_parameters
from qibolab._core.platform.platform import PARAMETERS
from qibolab._core.pulses.pulse import Pulse, Readout
from qibolab.platform import Hardware, initialize_parameters


def test_parameters_initialization():
    dummy = create_platform("dummy")

    hardware = Hardware(
        instruments=dummy.instruments,
        qubits=dummy.qubits,
        couplers=dummy.couplers,
    )
    hardware.instruments["dummy"].channels["0/drive"] = IqChannel(
        mixer="mixer/ciao", lo="lo/come"
    )
    parameters = initialize_parameters(
        hardware=hardware, natives=["RX", "RX12", "MZ", "CZ"], pairs=["0-2"]
    )

    for q in dummy.qubits:
        assert f"{q}/drive" in parameters.configs
        assert f"{q}/probe" in parameters.configs
        assert f"{q}/acquisition" in parameters.configs
        assert f"{q}/drive12" in parameters.configs
        assert q in parameters.native_gates.single_qubit
    for c in dummy.couplers:
        assert f"coupler_{c}/flux" in parameters.configs
        assert c in parameters.native_gates.coupler

    assert list(parameters.native_gates.two_qubit) == [(0, 2)]
    sequence = parameters.native_gates.two_qubit[(0, 2)].CZ
    assert sequence[0][0] == "0/flux"
    assert isinstance(sequence[0][1], Pulse)
    sequence = parameters.native_gates.single_qubit[0].RX
    assert sequence[0][0] == "0/drive"
    assert isinstance(sequence[0][1], Pulse)
    sequence = parameters.native_gates.single_qubit[2].MZ
    assert sequence[0][0] == "2/acquisition"
    assert isinstance(sequence[0][1], Readout)

    assert isinstance(parameters.configs["0/drive"], IqConfig)
    assert isinstance(parameters.configs["mixer/ciao"], IqMixerConfig)
    assert isinstance(parameters.configs["lo/come"], OscillatorConfig)


def test_parameters_reset(dummy_hardware: str):
    # ensure no parameters
    path = locate_platform(dummy_hardware)
    (path / PARAMETERS).unlink(missing_ok=True)

    with pytest.raises(FileNotFoundError):
        create_platform(dummy_hardware)

    # reset parameters
    reset_parameters(dummy_hardware, natives=["RX"])

    # load platform
    platform = create_platform(dummy_hardware)
    assert all(q.RX is not None for q in platform.natives.single_qubit.values())
    assert all(q.MZ is None for q in platform.natives.single_qubit.values())
