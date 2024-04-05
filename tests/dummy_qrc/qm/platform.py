import pathlib

from qibolab.instruments.dummy import DummyLocalOscillator as LocalOscillator
from qibolab.instruments.instrument_channel import ChannelMap, InstrumentChannel
from qibolab.instruments.qm import OPXplus, QMController
from qibolab.platform import Platform
from qibolab.serialize import (
    load_instrument_settings,
    load_qubits,
    load_runcard,
    load_settings,
)

FOLDER = pathlib.Path(__file__).parent


def create():
    """Dummy platform using Quantum Machines (QM) OPXs and Rohde Schwarz local
    oscillators.

    Based on QuantWare 5-qubit device.

    Used in ``test_instruments_qm.py`` and ``test_instruments_qmsim.py``
    """
    opxs = [OPXplus(f"con{i}") for i in range(1, 4)]
    controller = QMController("qm", "192.168.0.101:80", opxs=opxs, time_of_flight=280)

    # Create channel objects and map controllers to channels
    channels = ChannelMap()
    # readout
    channels |= InstrumentChannel(
        "L3-25_a", port=controller.ports((("con1", 10), ("con1", 9)))
    )
    channels |= InstrumentChannel(
        "L3-25_b", port=controller.ports((("con2", 10), ("con2", 9)))
    )
    # feedback
    channels |= InstrumentChannel(
        "L2-5_a", port=controller.ports((("con1", 2), ("con1", 1)), output=False)
    )
    channels |= InstrumentChannel(
        "L2-5_b", port=controller.ports((("con2", 2), ("con2", 1)), output=False)
    )
    # drive
    channels |= (
        InstrumentChannel(
            f"L3-1{i}", port=controller.ports((("con1", 2 * i), ("con1", 2 * i - 1)))
        )
        for i in range(1, 5)
    )
    channels |= InstrumentChannel(
        "L3-15", port=controller.ports((("con3", 2), ("con3", 1)))
    )
    # flux
    channels |= (
        InstrumentChannel(f"L4-{i}", port=opxs[1].ports(i)) for i in range(1, 6)
    )
    # TWPA
    channels |= "L4-26"

    # Instantiate local oscillators
    local_oscillators = [
        LocalOscillator("lo_readout_a", "192.168.0.39"),
        LocalOscillator("lo_readout_b", "192.168.0.31"),
        LocalOscillator("lo_drive_low", "192.168.0.32"),
        LocalOscillator("lo_drive_mid", "192.168.0.33"),
        LocalOscillator("lo_drive_high", "192.168.0.34"),
        LocalOscillator("twpa_a", "192.168.0.35"),
    ]
    # Map LOs to channels
    channels["L3-25_a"].local_oscillator = local_oscillators[0]
    channels["L3-25_b"].local_oscillator = local_oscillators[1]
    channels["L3-15"].local_oscillator = local_oscillators[2]
    channels["L3-11"].local_oscillator = local_oscillators[2]
    channels["L3-12"].local_oscillator = local_oscillators[3]
    channels["L3-13"].local_oscillator = local_oscillators[4]
    channels["L3-14"].local_oscillator = local_oscillators[4]
    channels["L4-26"].local_oscillator = local_oscillators[5]

    # create qubit objects
    runcard = load_runcard(FOLDER)
    qubits, couplers, pairs = load_qubits(runcard)

    # assign channels to qubits
    for q in [0, 1]:
        qubits[q].readout = channels["L3-25_a"]
        qubits[q].feedback = channels["L2-5_a"]
    for q in [2, 3, 4]:
        qubits[q].readout = channels["L3-25_b"]
        qubits[q].feedback = channels["L2-5_b"]

    qubits[0].drive = channels["L3-15"]
    qubits[0].flux = channels["L4-5"]
    for q in range(1, 5):
        qubits[q].drive = channels[f"L3-{10 + q}"]
        qubits[q].flux = channels[f"L4-{q}"]

    instruments = {controller.name: controller}
    instruments.update(controller.opxs)
    instruments.update({lo.name: lo for lo in local_oscillators})
    settings = load_settings(runcard)
    instruments = load_instrument_settings(runcard, instruments)
    return Platform("qm", qubits, pairs, instruments, settings, resonator_type="2D")
