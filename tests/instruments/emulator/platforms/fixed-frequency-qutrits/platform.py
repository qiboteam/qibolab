from qibolab import ConfigKinds, Hardware, IqChannel, Platform, Qubit
from qibolab.instruments.emulator import (
    DriveEmulatorConfig,
    EmulatorController,
    HamiltonianConfig,
)

ConfigKinds.extend([HamiltonianConfig, DriveEmulatorConfig])


def create() -> Platform:
    """Create a dummy platform using the dummy instrument."""
    qubits = {}
    channels = {}

    qubits[0] = qubit = Qubit.default(
        0, drive_extra={(1, 2): "0/drive12", 1: "01/drive"}
    )
    channels |= {
        qubit.drive: IqChannel(mixer=None, lo=None),
        qubits[0].drive_extra[1, 2]: IqChannel(mixer=None, lo=None),
        qubits[0].drive_extra[1]: IqChannel(mixer=None, lo=None),
    }
    qubits[1] = qubit = Qubit.default(1, drive_extra={(1, 2): "1/drive12"})
    channels |= {
        qubit.drive: IqChannel(mixer=None, lo=None),
        qubits[1].drive_extra[1, 2]: IqChannel(mixer=None, lo=None),
    }
    # register the instruments
    instruments = {
        "dummy": EmulatorController(address="0.0.0.0", channels=channels),
    }

    return Hardware(
        instruments=instruments,
        qubits=qubits,
    )
