import pathlib

from qibolab import (
    AcquisitionChannel,
    ConfigKinds,
    DcChannel,
    IqChannel,
    Platform,
    Qubit,
    QubitMap,
)
from qibolab.instruments.emulator import (
    DriveEmulatorConfig,
    EmulatorController,
    FluxEmulatorConfig,
    HamiltonianConfig,
)

FOLDER = pathlib.Path(__file__).parent

ConfigKinds.extend([HamiltonianConfig, DriveEmulatorConfig, FluxEmulatorConfig])


def create() -> Platform:
    """Create emulator platform with one qutrit."""
    qubits: QubitMap = {}
    channels = {}

    for q in range(1):
        qubits[q] = Qubit.default(q, drive_extra={(1, 2): f"{q}/drive12"})
        channels |= {
            qubits[q].acquisition: AcquisitionChannel(probe=qubits[q].probe),
            qubits[q].drive: IqChannel(mixer=None, lo=None),
            qubits[q].drive_extra[1, 2]: IqChannel(mixer=None, lo=None),
            qubits[q].flux: DcChannel(),
        }

    # register the instruments
    instruments = {
        "dummy": EmulatorController(address="0.0.0.0", channels=channels),
    }

    return Platform.load(
        path=FOLDER,
        instruments=instruments,
        qubits=qubits,
    )
