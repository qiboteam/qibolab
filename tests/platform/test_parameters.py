from qibolab._core.platform.load import create_platform
from qibolab._core.pulses.pulse import Pulse, Readout
from qibolab.platform import Hardware, initialize_parameters


def test_builder():
    dummy = create_platform("dummy")

    hardware = Hardware(
        instruments=dummy.instruments,
        qubits=dummy.qubits,
        couplers=dummy.couplers,
    )
    parameters = initialize_parameters(
        hardware=hardware, natives=["RX", "MZ", "CZ"], pairs=["0-2"]
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
