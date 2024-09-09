import pathlib

from qibolab.components import AcquisitionChannel, DcChannel, IqChannel
from qibolab.instruments.dummy import DummyInstrument, DummyLocalOscillator
from qibolab.platform import Platform
from qibolab.qubits import Qubit

FOLDER = pathlib.Path(__file__).parent


def create_dummy() -> Platform:
    """Create a dummy platform using the dummy instrument."""
    pump_name = "twpa_pump"

    qubits = {}
    channels = {}
    # attach the channels
    for q in range(5):
        drive, drive12, flux, probe, acquisition = (
            f"qubit_{q}/drive",
            f"qubit_{q}/drive12",
            f"qubit_{q}/flux",
            f"qubit_{q}/probe",
            f"qubit_{q}/acquisition",
        )
        channels |= {
            probe: IqChannel(mixer=None, lo=None),
            acquisition: AcquisitionChannel(twpa_pump=pump_name, probe=probe),
            drive: IqChannel(mixer=None, lo=None),
            drive12: IqChannel(mixer=None, lo=None),
            flux: DcChannel(),
        }
        qubits[q] = Qubit(
            probe=probe,
            acquisition=acquisition,
            drive=drive,
            drive_qudits={(1, 2): drive12},
            flux=flux,
        )

    couplers = {}
    for c in (0, 1, 3, 4):
        flux = f"coupler_{c}/flux"
        channels |= {flux: DcChannel()}
        couplers[c] = Qubit(flux=flux)

    # register the instruments
    instrument = DummyInstrument(name="dummy", address="0.0.0.0", channels=channels)
    pump = DummyLocalOscillator(name=pump_name, address="0.0.0.0")

    return Platform.load(
        path=FOLDER, instruments=[instrument, pump], qubits=qubits, couplers=couplers
    )
