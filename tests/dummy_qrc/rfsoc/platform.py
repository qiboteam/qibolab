import pathlib

from qibolab.channels import Channel, ChannelMap
from qibolab.instruments.erasynth import ERA
from qibolab.instruments.rfsoc import RFSoC
from qibolab.instruments.rohde_schwarz import SGS100A
from qibolab.platform import Platform
from qibolab.serialize import (
    load_instrument_settings,
    load_qubits,
    load_runcard,
    load_settings,
)

FOLDER = pathlib.Path(__file__).parent


def create(folder: pathlib.Path = FOLDER):
    """Dummy platform using QICK project on the RFSoC4x2 board.

    Used in ``test_instruments_rfsoc.py``.
    """
    # Instantiate QICK instruments
    controller = RFSoC("tii_rfsoc4x2", "0.0.0.0", 0, sampling_rate=9.8304)

    # Create channel objects and map to instrument controllers
    channels = ChannelMap()
    channels |= Channel("L3-18_ro", port=controller.ports(0))  # readout (DAC)
    channels |= Channel("L2-RO", port=controller.ports(0))  # feedback (readout DAC)
    channels |= Channel("L3-18_qd", port=controller.ports(1))  # drive
    channels |= Channel("L2-22_qf", port=controller.ports(2))  # flux

    lo_twpa = SGS100A("twpa_a", "192.168.0.32")
    lo_era = ERA("ErasynthLO", "192.168.0.212", ethernet=True)
    channels["L3-18_ro"].local_oscillator = lo_era

    runcard = load_runcard(FOLDER)
    qubits, couplers, pairs = load_qubits(runcard)

    # assign channels to qubits
    qubits[0].readout = channels["L3-18_ro"]
    qubits[0].feedback = channels["L2-RO"]
    qubits[0].drive = channels["L3-18_qd"]
    qubits[0].flux = channels["L2-22_qf"]

    instruments = {inst.name: inst for inst in [controller, lo_twpa, lo_era]}
    settings = load_settings(runcard)
    instruments = load_instrument_settings(runcard, instruments)
    return Platform(
        str(FOLDER), qubits, pairs, instruments, settings, resonator_type="3D"
    )
