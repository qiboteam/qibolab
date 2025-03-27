import pathlib

from qibolab.channels import Channel, ChannelMap
from qibolab.instruments.rfsoc import RFSoC
from qibolab.platform import Platform
from qibolab.serialize import load_qubits, load_runcard, load_settings

ADDRESS = "192.168.0.1"
PORT = 6000
FOLDER = pathlib.Path(__file__).parent


def create():
    # Instantiate QICK instruments
    controller = RFSoC(str(FOLDER), ADDRESS, PORT, sampling_rate=9.8304)
    controller.cfg.adc_trig_offset = 200
    controller.cfg.repetition_duration = 70
    # Create channel objects
    channels = ChannelMap()
    channels |= Channel("xxx", port=controller.ports(1))  # readout (DAC)
    channels |= Channel("yyy", port=controller.ports(0))  # feedback (readout ADC)
    channels |= Channel("zzz", port=controller.ports(0))  # drive

    # create qubit objects
    runcard = load_runcard(FOLDER)
    qubits, couplers, pairs = load_qubits(runcard)
    # assign channels to qubits
    qubits[0].readout = channels["xxx"]
    qubits[0].feedback = channels["yyy"]
    qubits[0].drive = channels["zzz"]

    instruments = {controller.name: controller}

    settings = load_settings(runcard)
    return Platform(
        str(FOLDER), qubits, pairs, instruments, settings, resonator_type="3D"
    )
