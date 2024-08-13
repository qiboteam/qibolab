import pathlib

from qibolab.components import AcquireChannel, DcChannel, IqChannel
from qibolab.instruments.dummy import DummyInstrument, DummyLocalOscillator
from qibolab.kernels import Kernels
from qibolab.parameters import Parameters
from qibolab.platform import Platform
from qibolab.serialize_ import replace

FOLDER = pathlib.Path(__file__).parent


def create_dummy():
    """Create a dummy platform using the dummy instrument."""
    instrument = DummyInstrument("dummy", "0.0.0.0")

    twpa_pump_name = "twpa_pump"
    twpa_pump = DummyLocalOscillator(twpa_pump_name, "0.0.0.0")

    parameters = Parameters.load(FOLDER)
    kernels = Kernels.load(FOLDER)

    configs = parameters.configs
    platform = Platform(
        FOLDER.name,
        parameters=parameters,
        configs=configs,
        instruments={instrument.name: instrument, twpa_pump.name: twpa_pump},
    )
    for q, qubit in platform.qubits.items():
        acquisition_name = f"qubit_{q}/acquire"
        probe_name = f"qubit_{q}/probe"
        qubit.probe = IqChannel(
            probe_name, mixer=None, lo=None, acquisition=acquisition_name
        )
        qubit.acquisition = AcquireChannel(
            acquisition_name, twpa_pump=twpa_pump_name, probe=probe_name
        )
        configs[acquisition_name] = replace(
            configs[acquisition_name], kernel=kernels.get(q)
        )

        drive_name = f"qubit_{q}/drive"
        qubit.drive = IqChannel(drive_name, mixer=None, lo=None, acquisition=None)

        drive_12_name = f"qubit_{q}/drive12"
        qubit.drive12 = IqChannel(drive_12_name, mixer=None, lo=None, acquisition=None)

        flux_name = f"qubit_{q}/flux"
        qubit.flux = DcChannel(flux_name)

    for c, coupler in platform.couplers.items():
        flux_name = f"coupler_{c}/flux"
        coupler.flux = DcChannel(flux_name)

    return platform


# TODO:
# "instruments": {
#   "dummy": {
#     "bounds": {
#       "waveforms": 0,
#       "readout": 0,
#       "instructions": 0
#     }
#   },
# },
