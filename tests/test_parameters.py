from typing import Literal

import pytest

from qibolab._core.components.configs import Config
from qibolab._core.native import Native, TwoQubitNatives
from qibolab._core.parameters import (
    ConfigKinds,
    Parameters,
    ParametersBuilder,
    TwoQubitContainer,
)
from qibolab._core.platform.load import create_platform
from qibolab._core.pulses.pulse import Pulse, Readout


def test_two_qubit_container():
    """The container guarantees access to symmetric interactions.

    Swapped indexing is working (only with getitem, not other dict
    methods) if all the registered natives are symmetric.
    """
    symmetric = TwoQubitContainer({(0, 1): TwoQubitNatives(CZ=Native())})
    assert symmetric[1, 0].CZ is not None

    asymmetric = TwoQubitContainer({(0, 1): TwoQubitNatives(CNOT=Native())})
    with pytest.raises(KeyError):
        asymmetric[(1, 0)]

    empty = TwoQubitContainer({(0, 1): TwoQubitNatives()})
    assert empty[(1, 0)] is not None


class DummyConfig(Config):
    kind: Literal["dummy"] = "dummy"
    ciao: str


class DummyConfig1(Config):
    kind: Literal["dummy1"] = "dummy1"
    come: int


class TestConfigKinds:
    # TODO: add @staticmethod and drop unused `self`, once py3.9 will be abandoned
    @pytest.fixture(autouse=True)
    def clean_kinds(self):
        ConfigKinds.reset()

    def test_manipulation(self):
        ConfigKinds.extend([DummyConfig])
        assert DummyConfig in ConfigKinds.registered()

        ConfigKinds.reset()
        assert DummyConfig not in ConfigKinds.registered()

        ConfigKinds.extend([DummyConfig, DummyConfig1])
        assert DummyConfig in ConfigKinds.registered()
        assert DummyConfig1 in ConfigKinds.registered()

    def test_adapted(self):
        ConfigKinds.extend([DummyConfig, DummyConfig1])
        adapted = ConfigKinds.adapted()

        dummy = DummyConfig(ciao="come")
        dump = adapted.dump_python(dummy)
        assert dump["ciao"] == "come"
        reloaded = adapted.validate_python(dump)
        assert reloaded == dummy

        dummy1 = DummyConfig1(come=42)
        dump1 = adapted.dump_python(dummy1)
        assert dump1["come"] == 42
        reloaded1 = adapted.validate_python(dump1)
        assert reloaded1 == dummy1

    def test_within_parameters(self):
        ConfigKinds.extend([DummyConfig, DummyConfig1])
        pars = Parameters(configs={"come": DummyConfig1(come=42)})

        dump = pars.model_dump()
        assert dump["configs"]["come"]["come"] == 42
        assert "dummy1" in pars.model_dump_json()

        reloaded = Parameters.model_validate(dump)
        assert reloaded == pars


def test_update():
    dummy = create_platform("dummy")
    dummy.update({})

    assert isinstance(dummy.parameters.native_gates.single_qubit[1].RX[0][1], Pulse)
    assert dummy.natives.single_qubit[1].RX[0][1].amplitude > 0
    dummy.update({"native_gates.single_qubit.1.RX.0.1.amplitude": -0.5})
    assert dummy.natives.single_qubit[1].RX[0][1].amplitude < 0

    assert dummy.settings.nshots != 1234567890
    dummy.update({"settings.nshots": 1234567890})
    assert dummy.settings.nshots == 1234567890

    dummy.update(
        {
            "settings.nshots": 42,
            "native_gates.single_qubit.1.RX.0.1.amplitude": -0.123,
            "native_gates.single_qubit.1.RX.0.1.duration": 456.7,
        }
    )
    assert dummy.settings.nshots == 42
    assert dummy.natives.single_qubit[1].RX[0][1].amplitude == -0.123
    assert dummy.natives.single_qubit[1].RX[0][1].duration == 456.7


def test_builder():
    dummy = create_platform("dummy")

    hardware = {
        "instruments": dummy.instruments,
        "qubits": dummy.qubits,
        "couplers": dummy.couplers,
    }
    builder = ParametersBuilder(
        hardware=hardware, natives=["RX", "MZ", "CZ"], pairs=["0-2"]
    )
    parameters = builder.build()

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
