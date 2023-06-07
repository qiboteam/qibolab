import pathlib

from qibolab.channels import ChannelMap
from qibolab.instruments.erasynth import ERA
from qibolab.platform import Platform

RUNCARD = pathlib.Path(__file__).parent / "rfsoc.yml"


def create(runcard=RUNCARD):
    """Dummy platform using QICK project on the RFSoC4x2 board.

    Used in ``test_instruments_rfsoc.py``.
    """
    from qibolab.instruments.rfsoc import RFSoC as TII_RFSOC4x2
    from qibolab.instruments.rohde_schwarz import SGS100A as LocalOscillator

    # Create channel objects
    channels = ChannelMap()
    channels |= "L3-18_ro"  # readout (DAC)
    channels |= "L2-RO"  # feedback (readout DAC)
    channels |= "L3-18_qd"  # drive
    channels |= "L2-22_qf"  # flux

    # Map controllers to qubit channels (HARDCODED)
    channels["L3-18_ro"].ports = [("o0", 0)]  # readout
    channels["L2-RO"].ports = [("i0", 0)]  # feedback
    channels["L3-18_qd"].ports = [("o1", 1)]  # drive
    channels["L2-22_qf"].ports = [("o2", 2)]  # flux

    local_oscillators = [
        LocalOscillator("twpa_a", "192.168.0.32"),
        ERA("ErasynthLO", "192.168.0.212", ethernet=True),
    ]
    local_oscillators[0].frequency = 6_200_000_000
    local_oscillators[0].power = -1

    local_oscillators[1].frequency = 0
    channels["L3-18_ro"].local_oscillator = local_oscillators[1]

    # Instantiate QICK instruments
    controller = TII_RFSOC4x2("tii_rfsoc4x2", "0.0.0.0", 0)
    instruments = [controller] + local_oscillators

    platform = Platform("tii_rfsoc4x2", RUNCARD, instruments, channels)

    # assign channels to qubits
    qubits = platform.qubits
    qubits[0].readout = channels["L3-18_ro"]
    qubits[0].feedback = channels["L2-RO"]
    qubits[0].drive = channels["L3-18_qd"]
    qubits[0].flux = channels["L2-22_qf"]

    channels["L2-22_qf"].qubit = qubits[0]

    return platform
