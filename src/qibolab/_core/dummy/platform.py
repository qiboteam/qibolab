import pathlib

from qibolab._core.components import AcquisitionChannel, DcChannel, IqChannel
from qibolab._core.instruments.dummy import DummyInstrument, DummyLocalOscillator
from qibolab._core.platform import Hardware, Platform
from qibolab._core.qubits import Qubit

FOLDER = pathlib.Path(__file__).parent


def create_dummy_hardware(
    qubits_set: list | None = None, couplers_set: list | None = None
) -> Hardware:
    """Create dummy hardware configuration based on the dummy instrument."""
    qubits = {}
    channels = {}
    # attach the channels
    pump_name = "twpa_pump"

    if qubits_set is None:
        qubits_set = list(range(5))
    if couplers_set is None:
        couplers_set = list(range(5))

    for q in qubits_set:
        drive12 = f"{q}/drive12"
        qubits[q] = qubit = Qubit.default(q, drive_extra={(1, 2): drive12})
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
    for c in couplers_set:
        couplers[c] = coupler = Qubit(flux=f"coupler_{c}/flux")
        channels |= {coupler.flux: DcChannel()}

    # register the instruments
    instruments = {
        "dummy": DummyInstrument(address="0.0.0.0", channels=channels),
        pump_name: DummyLocalOscillator(address="0.0.0.0"),
    }

    return Hardware(instruments=instruments, qubits=qubits, couplers=couplers)


def create_dummy_platform(
    qubits_set: list | None = None, couplers_set: list | None = None
) -> Platform:
    """Create a dummy platform using the dummy instrument."""
    hardware = create_dummy_hardware(qubits_set=qubits_set, couplers_set=couplers_set)
    return Platform.load(path=FOLDER, **vars(hardware))
