import pathlib

from qibolab.channels import ChannelMap
from qibolab.instruments.emulator.pulse_simulator import PulseSimulator
from qibolab.platform import Platform
from qibolab.serialize import (
    load_instrument_settings,
    load_qubits,
    load_runcard,
    load_settings,
)

FOLDER = pathlib.Path(__file__).parent


def create():
    """Create a multi-qubit emulator platform with flux pulses."""

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

    # Create channel object
    channels = ChannelMap()
    channels |= (f"readout-{q}" for q in qubits)
    channels |= (f"drive-{q}" for q in qubits)
    channels |= (f"flux-{q}" for q in qubits)

    # map channels to qubits
    for q, qubit in qubits.items():
        qubit.readout = channels[f"readout-{q}"]
        qubit.drive = channels[f"drive-{q}"]
        qubit.flux = channels[f"flux-{q}"]

        channels[f"drive-{q}"].qubit = qubit
        channels[f"flux-{q}"].qubit = qubit
        qubit.sweetspot = 0  # not used

    return Platform(
        device_name,
        qubits,
        pairs,
        instruments,
        settings,
        resonator_type="2D",
    )
