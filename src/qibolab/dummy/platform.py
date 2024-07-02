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
    load_component_config,
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

    component_configs = {}
    component_configs[twpa_pump_name] = load_component_config(
        runcard, twpa_pump_name, OscillatorConfig
    )
    for q, qubit in qubits.items():
        acquisition_name = f"qubit_{q}/acquire"
        measure_name = f"qubit_{q}/measure"
        qubit.measure = IqChannel(
            measure_name, mixer=None, lo=None, acquisition=acquisition_name
        )
        qubit.acquisition = AcquireChannel(
            acquisition_name, twpa_pump=twpa_pump_name, measure=measure_name
        )
        component_configs[measure_name] = load_component_config(
            runcard, measure_name, IqConfig
        )
        component_configs[acquisition_name] = load_component_config(
            runcard, acquisition_name, AcquisitionConfig
        )

        drive_name = f"qubit_{q}/drive"
        qubit.drive = IqChannel(drive_name, mixer=None, lo=None, acquisition=None)
        component_configs[drive_name] = load_component_config(
            runcard, drive_name, IqConfig
        )

        drive_12_name = f"qubit_{q}/drive12"
        qubit.drive12 = IqChannel(drive_12_name, mixer=None, lo=None, acquisition=None)
        component_configs[drive_12_name] = load_component_config(
            runcard, drive_12_name, IqConfig
        )

        flux_name = f"qubit_{q}/flux"
        qubit.flux = DcChannel(flux_name)
        component_configs[flux_name] = load_component_config(
            runcard, flux_name, DcConfig
        )

    if with_couplers:
        for c, coupler in couplers.items():
            flux_name = f"coupler_{c}/flux"
            coupler.flux = DcChannel(flux_name)
            component_configs[flux_name] = load_component_config(
                runcard, flux_name, DcConfig
            )

    instruments = {instrument.name: instrument, twpa_pump.name: twpa_pump}
    name = "dummy_couplers" if with_couplers else "dummy"
    return Platform(
        name,
        qubits,
        pairs,
        component_configs,
        instruments,
        settings,
        resonator_type="2D",
        couplers=couplers,
    )
