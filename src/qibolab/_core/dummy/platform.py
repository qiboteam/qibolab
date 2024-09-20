import pathlib

from qibolab._core.components import AcquisitionChannel, DcChannel, IqChannel
from qibolab._core.instruments.dummy import DummyInstrument, DummyLocalOscillator
from qibolab._core.platform import Platform
from qibolab._core.qubits import Qubit

FOLDER = pathlib.Path(__file__).parent


def create_dummy() -> Platform:
    """Create a dummy platform using the dummy instrument."""
    qubits = {}
    channels = {}
    # attach the channels
    pump_name = "twpa_pump"
    for q in range(5):
        drive12 = f"{q}/drive12"
        qubits[q] = qubit = Qubit.default(q, drive_qudits={(1, 2): drive12})
        channels |= {
            qubit.probe: IqChannel(mixer=None, lo=None),
            qubit.acquisition: AcquisitionChannel(
                twpa_pump=pump_name, probe=qubit.probe
            ),
            qubit.drive: IqChannel(mixer=None, lo=None),
            drive12: IqChannel(mixer=None, lo=None),
            qubit.flux: DcChannel(),
        }

    couplers = {}
    for c in (0, 1, 3, 4):
        couplers[c] = coupler = Qubit(flux=f"coupler_{c}/flux")
        channels |= {coupler.flux: DcChannel()}

    # register the instruments
    instruments = {
        "dummy": DummyInstrument(address="0.0.0.0", channels=channels),
        pump_name: DummyLocalOscillator(address="0.0.0.0"),
    }

    return Platform.load(
        path=FOLDER, instruments=instruments, qubits=qubits, couplers=couplers
    )
