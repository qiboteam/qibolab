import pathlib

from qibolab._core.components import AcquisitionChannel, DcChannel, IqChannel
from qibolab._core.instruments.dummy import DummyInstrument, DummyLocalOscillator
from qibolab._core.emulator.emulator import EmulatorController
from qibolab._core.emulator.model import QubitConfig
from qibolab._core.platform import Platform
from qibolab._core.qubits import Qubit
from qibolab import ConfigKinds

FOLDER = pathlib.Path(__file__).parent

ConfigKinds.extend([QubitConfig])

def create_emulator() -> Platform:
    """Create a dummy platform using the dummy instrument."""
    qubits = {}
    channels = {}

    for q in range(1):
        qubits[q] = qubit = Qubit.default(q)
        channels |= {
            qubit.drive: IqChannel(mixer=None, lo=None),
        }

    # need to pass physical information -> frequency of each qubit and anharmonicity
    # register the instruments
    instruments = {
        "dummy": EmulatorController(address="0.0.0.0", channels=channels),
    }

    return Platform.load(
        path=FOLDER, instruments=instruments, qubits=qubits,
    )