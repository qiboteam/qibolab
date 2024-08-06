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


def remove_couplers(runcard):
    """Remove coupler sections from runcard to create a dummy platform without
    couplers."""
    runcard["topology"] = list(runcard["topology"].values())
    del runcard["couplers"]
    del runcard["characterization"]["coupler"]
    two_qubit = runcard["native_gates"]["two_qubit"]
    for i, gates in two_qubit.items():
        for j, gate in gates.items():
            two_qubit[i][j] = {
                ch: pulses for ch, pulses in gate.items() if "coupler" not in ch
            }
    return runcard


def create_dummy(with_couplers: bool = True):
    """Create a dummy platform using the dummy instrument.

    Args:
        with_couplers (bool): Selects whether the dummy platform will have coupler qubits.
    """
    instrument = DummyInstrument("dummy", "0.0.0.0")

    twpa_pump_name = "twpa_pump"
    twpa_pump = DummyLocalOscillator(twpa_pump_name, "0.0.0.0")

    runcard = load_runcard(FOLDER)
    kernels = Kernels.load(FOLDER)

    if not with_couplers:
        runcard = remove_couplers(runcard)

    qubits, couplers, pairs = load_qubits(runcard, kernels)
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
            **component_params[acquisition_name]
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

    if with_couplers:
        for c, coupler in couplers.items():
            flux_name = f"coupler_{c}/flux"
            coupler.flux = DcChannel(flux_name)
            configs[flux_name] = DcConfig(**component_params[flux_name])

    instruments = {instrument.name: instrument, twpa_pump.name: twpa_pump}
    instruments = load_instrument_settings(runcard, instruments)
    name = "dummy_couplers" if with_couplers else "dummy"
    return Platform(
        name,
        qubits,
        pairs,
        configs,
        instruments,
        settings,
        resonator_type="2D",
        couplers=couplers,
    )
