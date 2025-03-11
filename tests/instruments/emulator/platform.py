import pathlib

from qibolab import ConfigKinds
from qibolab._core.components import IqChannel
from qibolab._core.instruments.emulator.emulator import EmulatorController
from qibolab._core.instruments.emulator.hamiltonians import HamiltonianConfig
from qibolab._core.platform import Platform
from qibolab._core.qubits import Qubit

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
