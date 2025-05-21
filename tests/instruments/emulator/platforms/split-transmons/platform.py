from qibolab import ConfigKinds, DcChannel, Hardware, IqChannel, Qubit
from qibolab.instruments.emulator import (
    DriveEmulatorConfig,
    EmulatorController,
    FluxEmulatorConfig,
    HamiltonianConfig,
)

ConfigKinds.extend([HamiltonianConfig, DriveEmulatorConfig, FluxEmulatorConfig])


def create() -> Hardware:
    """Create platform with two split-transmons coupled."""
    qubits = {}
    channels = {}

    for q in range(2):
        qubits[q] = qubit = Qubit.default(q, drive_extra={(1, 2): f"{q}/drive12"})
        channels |= {
            qubit.drive: IqChannel(mixer=None, lo=None),
            qubit.flux: DcChannel(),
            qubits[q].drive_extra[1, 2]: IqChannel(mixer=None, lo=None),
        }
    # register the instruments
    instruments = {
        "emulator": EmulatorController(address="0.0.0.0", channels=channels),
    }

    return Hardware(
        instruments=instruments,
        qubits=qubits,
    )
