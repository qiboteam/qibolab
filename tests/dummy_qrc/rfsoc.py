import pathlib

from qibolab.channels import Channel, ChannelMap
from qibolab.instruments.erasynth import ERA
from qibolab.instruments.rfsoc import RFSoC
from qibolab.instruments.rohde_schwarz import SGS100A
from qibolab.platform import Platform

RUNCARD = pathlib.Path(__file__).parent / "rfsoc.yml"


def create(runcard=RUNCARD):
    """Dummy platform using QICK project on the RFSoC4x2 board.

    Used in ``test_instruments_rfsoc.py``.
    """
    # Instantiate QICK instruments
    controller = RFSoC("tii_rfsoc4x2", "0.0.0.0", 0)

    # Create channel objects and map to instrument controllers
    channels = ChannelMap()
    channels |= Channel("L3-18_ro", port=controller[0])  # readout (DAC)
    channels |= Channel("L2-RO", port=controller[0])  # feedback (readout DAC)
    channels |= Channel("L3-18_qd", port=controller[1])  # drive
    channels |= Channel("L2-22_qf", port=controller[2])  # flux

    local_oscillators = [
        SGS100A("twpa_a", "192.168.0.32"),
        ERA("ErasynthLO", "192.168.0.212", ethernet=True),
    ]
    local_oscillators[0].frequency = 6_200_000_000
    local_oscillators[0].power = -1

    local_oscillators[1].frequency = 0
    channels["L3-18_ro"].local_oscillator = local_oscillators[1]

    instruments = [controller] + local_oscillators

    platform = Platform("tii_rfsoc4x2", RUNCARD, instruments, channels)

    # assign channels to qubits
    qubits = platform.qubits
    qubits[0].readout = channels["L3-18_ro"]
    qubits[0].feedback = channels["L2-RO"]
    qubits[0].drive = channels["L3-18_qd"]
    qubits[0].flux = channels["L2-22_qf"]

    return platform
