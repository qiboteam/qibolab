import pathlib

from qibolab import AcquisitionChannel, DcChannel, IqChannel, Qubit
from qibolab.instruments.era import ERASynth
from qibolab.instruments.rfsoc import RFSoC
from qibolab.instruments.rohde_schwarz import SGS100A
from qibolab.platform import Platform

FOLDER = pathlib.Path(__file__).parent


def create():
    """Dummy platform using QICK project on the RFSoC4x2 board.

    Used in ``test_instruments_rfsoc.py``.
    """

    qubit = Qubit.default("q0")

    # offset?

    channels = {}

    # Readout
    assert qubit.probe is not None
    channels[qubit.probe] = IqChannel(
        device="L3-18_ro", path="0", mixer=None, lo="ErasynthLO"
    )

    # Acquire (feedback)
    assert qubit.acquisition is not None
    channels[qubit.acquisition] = AcquisitionChannel(
        device="L2-RO",
        path="0",
        probe=qubit.probe,  # twpa_pump="twpa_a" ?
    )

    # Drive
    assert qubit.drive is not None
    channels[qubit.drive] = IqChannel(
        device="L3-18_qd",
        path="1",
        mixer=None,  # lo="ErasynthLO" ?
    )

    # Flux
    assert qubit.flux is not None
    channels[qubit.flux] = DcChannel(device="L2-22_qf", path="2")

    lo_twpa = SGS100A(address="192.168.0.32")
    lo_era = ERASynth(address="192.168.0.212", ethernet=True)
    controller = RFSoC(
        address="0.0.0.0",
        channels=channels,
        port=0,
    )

    instruments = {"tii_rfsoc4x2": controller, "twpa_a": lo_twpa, "ErasynthLO": lo_era}
    return Platform.load(path=FOLDER, instruments=instruments, qubits={"q0": qubit})
