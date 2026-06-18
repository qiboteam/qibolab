from pathlib import Path

from qibolab import AcquisitionChannel, ConfigKinds, Hardware, IqChannel, Qubit
from qibolab.instruments.emulator import (
    DriveEmulatorConfig,
    EmulatorController,
    HamiltonianConfig,
)

ConfigKinds.extend([HamiltonianConfig, DriveEmulatorConfig])


def create() -> Hardware:
    """Create emulator platform with fixed-frequency coupled qutrits."""
    qubits = {}
    channels = {}

    qubits[0] = qubit = Qubit.default(
        0, drive_extra={(1, 2): "0/drive12", 1: "0/drive1"}
    )
    channels |= {
        qubit.acquisition: AcquisitionChannel(probe=qubit.probe),
        qubit.drive: IqChannel(mixer=None, lo=None),
        qubit.drive_extra[1, 2]: IqChannel(mixer=None, lo=None),
        qubit.drive_extra[1]: IqChannel(mixer=None, lo=None),
    }
    qubits[1] = qubit = Qubit.default(1, drive_extra={(1, 2): "1/drive12"})
    channels |= {
        qubit.acquisition: AcquisitionChannel(probe=qubit.probe),
        qubit.drive: IqChannel(mixer=None, lo=None),
        qubit.drive_extra[1, 2]: IqChannel(mixer=None, lo=None),
    }

    dump_dir = "qutip_data"
    counter = 0
    directory = Path.cwd() / f"{dump_dir}_{counter}"
    while directory.exists():
        directory = Path.cwd() / f"{dump_dir}_{counter}"
        counter += 1

    # register the instruments
    instruments = {
        "emulator": EmulatorController(
            address="0.0.0.0",
            channels=channels,
            sampling_rate_=2,
            save_dir="./emulator_dumping_test_1",
        ),
    }

    return Hardware(
        instruments=instruments,
        qubits=qubits,
    )
