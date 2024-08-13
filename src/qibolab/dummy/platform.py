import pathlib

from qibolab.components import AcquireChannel, DcChannel, IqChannel
from qibolab.instruments.dummy import DummyInstrument, DummyLocalOscillator
from qibolab.platform import Platform

FOLDER = pathlib.Path(__file__).parent


def create_dummy():
    """Create a dummy platform using the dummy instrument."""
    # register the instruments
    instrument = DummyInstrument("dummy", "0.0.0.0")
    twpa_pump = DummyLocalOscillator("twpa_pump", "0.0.0.0")

    platform = Platform.load(path=FOLDER, instruments=[instrument, twpa_pump])

    # attach the channels
    for q, qubit in platform.qubits.items():
        acquisition_name = f"qubit_{q}/acquire"
        probe_name = f"qubit_{q}/probe"
        qubit.probe = IqChannel(
            probe_name, mixer=None, lo=None, acquisition=acquisition_name
        )
        qubit.acquisition = AcquireChannel(
            acquisition_name, twpa_pump=twpa_pump.name, probe=probe_name
        )

        qubit.drive = IqChannel(f"qubit_{q}/drive", mixer=None, lo=None)
        qubit.drive12 = IqChannel(f"qubit_{q}/drive12", mixer=None, lo=None)
        qubit.flux = DcChannel(f"qubit_{q}/flux")

    for c, coupler in platform.couplers.items():
        coupler.flux = DcChannel(f"coupler_{c}/flux")

    return platform
