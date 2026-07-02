import pathlib

from qibolab import (
    AcquisitionChannel,
    ConfigKinds,
    DcChannel,
    IqChannel,
    Platform,
    Qubit,
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
    qubits = {}
    channels = {}

    for q in range(1):
        qubits[q] = qubit = Qubit.default(q)
        channels |= {
            qubit.drive: IqChannel(mixer=None, lo=None),
            qubit.flux: DcChannel(),
            qubit.acquisition: AcquisitionChannel(probe=qubit.probe),
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
