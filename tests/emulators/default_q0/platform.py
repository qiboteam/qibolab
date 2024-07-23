import pathlib

from qibolab.components import IqChannel, AcquireChannel
from qibolab.instruments.emulator.pulse_simulator import PulseSimulator
from qibolab.platform import Platform
from qibolab.serialize import (
    load_component_config,
    load_instrument_settings,
    load_qubits,
    load_runcard,
    load_settings,
)

FOLDER = pathlib.Path(__file__).parent


def create():
    """Create a one qubit emulator platform."""

    # load runcard and model params
    runcard = load_runcard(FOLDER)
    device_name = runcard["device_name"]

    # Specify emulator controller
    pulse_simulator = PulseSimulator()
    instruments = {"pulse_simulator": pulse_simulator}
    instruments = load_instrument_settings(runcard, instruments)

    # extract quantities from runcard for platform declaration
    qubits, couplers, pairs = load_qubits(runcard)
    settings = load_settings(runcard)

    # define channels for qubits
    for q, qubit in qubits.items():
        qubit.measure = IqChannel("measure-{q}", mixer=None, lo=None, acquisition="acquire-{q}")
        qubit.acquisition = AcquireChannel("acquire-{q}", mixer=None, lo=None, measure="measure-{q}")

    return Platform(
        device_name,
        qubits,
        pairs,
        {},
        instruments,
        settings,
        resonator_type="2D",
    )
