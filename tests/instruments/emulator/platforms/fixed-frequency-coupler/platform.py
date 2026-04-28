from qibolab import ConfigKinds, DcChannel, Hardware, IqChannel, Qubit
from qibolab.instruments.emulator import (
    DriveEmulatorConfig,
    EmulatorController,
    HamiltonianConfig,
)

ConfigKinds.extend([HamiltonianConfig, DriveEmulatorConfig])


def create() -> Hardware:
    """Create emulator platform with fixed-frequency coupled qutrits."""
    qubits = {}
    couplers = {}
    channels = {}

    qubits[0] = qubit = Qubit.default(
        0, drive_extra={(1, 2): "0/drive12", 1: "0/drive1"}
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

    couplers[2] = Qubit.coupler(2, conn_qubits=(0, 1))
    channels |= {couplers[2].flux: DcChannel()}

    # register the instruments
    instruments = {
        "emulator": EmulatorController(
            address="0.0.0.0",
            channels=channels,
            sampling_rate_=2,
        ),
    }

    return Hardware(instruments=instruments, qubits=qubits, couplers=couplers)
