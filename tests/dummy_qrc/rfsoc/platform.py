import pathlib

from qibolab import AcquisitionChannel, DcChannel, IqChannel, Platform, Qubit
from qibolab.instruments.era import ERASynth

# from qibolab.instruments.rfsoc import RFSoC
from qibolab.instruments.rohde_schwarz import SGS100A

FOLDER = pathlib.Path(__file__).parent


def create():
    """Dummy platform using QICK project on the RFSoC4x2 board.

    Used in ``test_instruments_rfsoc.py``.
    """

    qubit = Qubit(
        probe="0/L3-18_ro", acquisition="0/L2-RO", drive="0/L3-18_qd", flux="0/L2-22_qf"
    )

    # offset?

    channels = {}

    # Readout (probe)
    assert qubit.probe is not None
    channels[qubit.probe] = IqChannel(
        device="0/L3-18_ro", path="0", mixer=None, lo="ErasynthLO"
    )

    # Acquire (feedback)
    assert qubit.acquisition is not None
    channels[qubit.acquisition] = AcquisitionChannel(
        device="0/L2-RO",
        path="0",
        probe=qubit.probe,
        twpa_pump=None,  # "twpa_a" ?
    )

    # Drive
    assert qubit.drive is not None
    channels[qubit.drive] = IqChannel(
        device="0/L3-18_qd",
        path="1",
        mixer=None,
        lo=None,  # "ErasynthLO" ?
    )

    # Flux
    assert qubit.flux is not None
    channels[qubit.flux] = DcChannel(device="0/L2-22_qf", path="2")

    lo_twpa = SGS100A(address="192.168.0.32")
    lo_era = ERASynth(address="192.168.0.212", ethernet=True)
    controller = lo_era  # RFSoC(
    #     address="0.0.0.0",
    #     channels=channels,
    #     port=0,
    # )

    instruments = {"tii_rfsoc4x2": controller, "twpa_a": lo_twpa, "ErasynthLO": lo_era}
    return Platform.load(path=FOLDER, instruments=instruments, qubits={0: qubit})
