import pathlib

from qibolab import ConfigKinds, IqChannel, Platform, Qubit
from qibolab.instruments.emulator import EmulatorController, HamiltonianConfig

FOLDER = pathlib.Path(__file__).parent

ConfigKinds.extend([HamiltonianConfig])


def create() -> Platform:
    """Create a dummy platform using the dummy instrument."""
    qubits = {}
    channels = {}

    for q in range(1):
        qubits[q] = qubit = Qubit.default(q)
        channels |= {
            qubit.drive: IqChannel(mixer=None, lo=None),
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
