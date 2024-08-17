import pathlib

from qibolab.components import AcquireChannel, DcChannel, IqChannel
from qibolab.instruments.dummy import DummyInstrument, DummyLocalOscillator
from qibolab.platform import Platform
from qibolab.qubits import Qubit

FOLDER = pathlib.Path(__file__).parent


def create_dummy() -> Platform:
    """Create a dummy platform using the dummy instrument."""
    # register the instruments
    instrument = DummyInstrument("dummy", "0.0.0.0")
    pump = DummyLocalOscillator("twpa_pump", "0.0.0.0")

    qubits = {}
    # attach the channels
    for q in range(5):
        probe, acquisition = f"qubit_{q}/probe", f"qubit_{q}/acquire"
        qubits[q] = Qubit(
            name=q,
            probe=IqChannel(probe, mixer=None, lo=None, acquisition=acquisition),
            acquisition=AcquireChannel(acquisition, twpa_pump=pump.name, probe=probe),
            drive=IqChannel(f"qubit_{q}/drive", mixer=None, lo=None),
            drive12=IqChannel(f"qubit_{q}/drive12", mixer=None, lo=None),
            flux=DcChannel(f"qubit_{q}/flux"),
        )

    couplers = {}
    for c in (0, 1, 3, 4):
        couplers[c] = Qubit(name=c, flux=DcChannel(f"coupler_{c}/flux"))

    return Platform.load(
        path=FOLDER, instruments=[instrument, pump], qubits=qubits, couplers=couplers
    )
