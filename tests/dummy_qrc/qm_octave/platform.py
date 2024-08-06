import pathlib

from qibolab.channel import Channel, ChannelMap
from qibolab.instruments.dummy import DummyLocalOscillator as LocalOscillator
from qibolab.instruments.qm import Octave, OPXplus, QMController
from qibolab.platform import Platform
from qibolab.serialize import (
    load_instrument_settings,
    load_qubits,
    load_runcard,
    load_settings,
)

RUNCARD = pathlib.Path(__file__).parent


def create(runcard_path=RUNCARD):
    """Dummy platform using Quantum Machines (QM) OPXs and Octaves.

    Based on QuantWare 5-qubit device.

    Used in ``test_instruments_qm.py`` and ``test_instruments_qmsim.py``
    """
    opxs = [OPXplus(f"con{i}") for i in range(1, 4)]
    octave1 = Octave("octave1", port=100, connectivity=opxs[0])
    octave2 = Octave("octave2", port=101, connectivity=opxs[1])
    octave3 = Octave("octave3", port=102, connectivity=opxs[2])
    controller = QMController(
        "qm",
        "192.168.0.101:80",
        opxs=opxs,
        octaves=[octave1, octave2, octave3],
        time_of_flight=280,
    )

    # Create channel objects and map controllers to channels
    channels = ChannelMap()
    # readout
    channels |= Channel("L3-25_a", port=octave1.ports(5))
    channels |= Channel("L3-25_b", port=octave2.ports(5))
    # feedback
    channels |= Channel("L2-5_a", port=octave1.ports(1, output=False))
    channels |= Channel("L2-5_b", port=octave2.ports(1, output=False))
    # drive
    channels |= (Channel(f"L3-1{i}", port=octave1.ports(i)) for i in range(1, 5))
    channels |= Channel("L3-15", port=octave3.ports(1))
    # flux
    channels |= (Channel(f"L4-{i}", port=opxs[1].ports(i)) for i in range(1, 6))
    # TWPA
    channels |= "L4-26"

    # Instantiate local oscillators
    twpa = LocalOscillator("twpa_a", "192.168.0.35")
    # Map LOs to channels
    channels["L4-26"].local_oscillator = twpa

    # create qubit objects
    runcard = load_runcard(runcard_path)
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

    instruments = {controller.name: controller, twpa.name: twpa}
    instruments.update(controller.opxs)
    instruments.update(controller.octaves)
    settings = load_settings(runcard)
    instruments = load_instrument_settings(runcard, instruments)
    return Platform(
        "qm_octave", qubits, pairs, instruments, settings, resonator_type="2D"
    )
