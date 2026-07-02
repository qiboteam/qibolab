from qibolab import (
    AcquisitionChannel,
    ConfigKinds,
    DcChannel,
    Hardware,
    IqChannel,
    Qubit,
    QubitMap,
)
from qibolab.instruments.emulator import (
    DriveEmulatorConfig,
    EmulatorController,
    FluxEmulatorConfig,
    HamiltonianConfig,
)

ConfigKinds.extend([HamiltonianConfig, DriveEmulatorConfig, FluxEmulatorConfig])


def create() -> Hardware:
    """Create platform with two split-transmons coupled."""
    qubits: QubitMap = {}
    channels = {}

    for q in range(2):
        qubits[q] = Qubit.default(q, drive_extra={(1, 2): f"{q}/drive12"})
        channels |= {
            qubits[q].acquisition: AcquisitionChannel(probe=qubits[q].probe),
            qubits[q].drive: IqChannel(mixer=None, lo=None),
            qubits[q].flux: DcChannel(),
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
