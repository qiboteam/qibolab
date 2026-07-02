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
    """Create emulator platform with one qubit."""
    qubits: QubitMap = {}
    channels = {}

    for q in range(1):
        qubits[q] = Qubit.default(q)
        channels |= {
            qubits[q].drive: IqChannel(mixer=None, lo=None),
            qubits[q].flux: DcChannel(),
            qubits[q].acquisition: AcquisitionChannel(probe=qubits[q].probe),
        }

    # register the instruments
    instruments = {
        "dummy": EmulatorController(
            address="0.0.0.0", channels=channels, sampling_rate_=1
        ),
    }

    return Platform.load(
        path=FOLDER,
        instruments=instruments,
        qubits=qubits,
    )
