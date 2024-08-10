import pathlib

from qibolab.components import (
    AcquireChannel,
    AcquisitionConfig,
    DcChannel,
    DcConfig,
    IqChannel,
    IqConfig,
    OscillatorConfig,
)
from qibolab.instruments.dummy import DummyInstrument, DummyLocalOscillator
from qibolab.kernels import Kernels
from qibolab.platform import Platform
from qibolab.serialize import (
    load_instrument_settings,
    load_qubits,
    load_runcard,
    load_settings,
)

FOLDER = pathlib.Path(__file__).parent


def create_dummy():
    """Create a dummy platform using the dummy instrument."""
    instrument = DummyInstrument("dummy", "0.0.0.0")

    twpa_pump_name = "twpa_pump"
    twpa_pump = DummyLocalOscillator(twpa_pump_name, "0.0.0.0")

    runcard = load_runcard(FOLDER)
    kernels = Kernels.load(FOLDER)

    qubits, couplers, pairs = load_qubits(runcard)
    settings = load_settings(runcard)

    configs = {}
    component_params = runcard["components"]
    configs[twpa_pump_name] = OscillatorConfig(**component_params[twpa_pump_name])
    for q, qubit in qubits.items():
        acquisition_name = f"qubit_{q}/acquire"
        probe_name = f"qubit_{q}/probe"
        qubit.probe = IqChannel(
            probe_name, mixer=None, lo=None, acquisition=acquisition_name
        )
        qubit.acquisition = AcquireChannel(
            acquisition_name, twpa_pump=twpa_pump_name, probe=probe_name
        )
        configs[probe_name] = IqConfig(**component_params[probe_name])
        configs[acquisition_name] = AcquisitionConfig(
            **component_params[acquisition_name], kernel=kernels.get(q)
        )

        drive_name = f"qubit_{q}/drive"
        qubit.drive = IqChannel(drive_name, mixer=None, lo=None, acquisition=None)
        configs[drive_name] = IqConfig(**component_params[drive_name])

        drive_12_name = f"qubit_{q}/drive12"
        qubit.drive12 = IqChannel(drive_12_name, mixer=None, lo=None, acquisition=None)
        configs[drive_12_name] = IqConfig(**component_params[drive_12_name])

        flux_name = f"qubit_{q}/flux"
        qubit.flux = DcChannel(flux_name)
        configs[flux_name] = DcConfig(**component_params[flux_name])

    for c, coupler in couplers.items():
        flux_name = f"coupler_{c}/flux"
        coupler.flux = DcChannel(flux_name)
        configs[flux_name] = DcConfig(**component_params[flux_name])

    instruments = {instrument.name: instrument, twpa_pump.name: twpa_pump}
    instruments = load_instrument_settings(runcard, instruments)
    return Platform(
        FOLDER.name,
        qubits,
        pairs,
        configs,
        instruments,
        settings,
        resonator_type="2D",
        couplers=couplers,
    )
